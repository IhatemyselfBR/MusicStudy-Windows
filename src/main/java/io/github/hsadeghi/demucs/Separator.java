// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Chunked separation on top of the exported graph.
 *
 * <p>The chunk offsets, the triangular cross-fade and the global normalisation are demucs' own, so
 * the output matches {@code apply_model(split=True, shifts=0)}. What differs is the scheduling:
 * consecutive chunks are batched into one inference call (upstream feeds them one at a time, which
 * leaves a GPU idle), the next batch's spectrogram is built while the current one is in the
 * network, and output is streamed out as it is finalised.
 *
 * <p>Bags of models ({@code htdemucs_ft}, {@code mdx}, {@code mdx_extra}, ...) are handled here
 * too: every member sees the same chunk, and their {@code spec} and {@code wav} outputs are
 * blended with the manifest's per-stem weights before the single ISTFT. Upstream instead runs
 * each member over the whole track and adds up the finished stems; the results agree because the
 * ISTFT, the cross-fade and the denormalisation are all linear, and this way the expensive half
 * is done once.
 */
public final class Separator implements AutoCloseable {

    /** Everything tunable about a separation run. */
    public static final class Options {
        public final SessionFactory.Config session = new SessionFactory.Config();
        /** Chunks per inference call; 0 means 1 on CPU and 4 on an accelerator. */
        public int batch = 0;
        /** demucs' default is 0.25. Lower is faster and rougher at the seams. */
        public float overlap = 0.25f;
        public float transitionPower = 1.0f;
        /** Overlap spectrogram work with inference. */
        public boolean pipeline = true;
    }

    /** Timing report from {@link #benchmark(double)}. */
    public record Benchmark(SessionFactory.Device device, int batch, int models,
                            double audioSeconds, double elapsed) {
        public double realtimeFactor() {
            return audioSeconds / elapsed;
        }
    }

    private final OrtEnvironment environment = OrtEnvironment.getEnvironment();
    private final List<OrtSession> sessions;
    /** Row-major {@code [model][source]} blend weights; null unless this is a bag. */
    private final float[][] mixing;
    private final SessionFactory.Device device;
    private final ModelConfig config;
    private final Spectrogram spectrogram;
    private final Options options;
    private final int batch;
    private final float[] weight;
    private final ExecutorService prefetch;
    private boolean checkedFinite;
    /** One member's answer, before it is folded into the blend. Null for a lone graph. */
    private float[] partSpec;
    private float[] partWav;

    public Separator(Path modelPath, Options options) throws OrtException {
        this.options = options;
        ModelBundle bundle = ModelBundle.resolve(modelPath);

        List<OrtSession> opened = new ArrayList<>(bundle.members().size());
        SessionFactory.Device chosen = SessionFactory.Device.CPU;
        ModelConfig first = null;
        for (ModelBundle.Member member : bundle.members()) {
            SessionFactory.Opened result = SessionFactory.create(member.path(), options.session);
            opened.add(result.session());
            chosen = result.device();

            ModelConfig theirs = ModelConfig.fromSession(result.session(), member.path());
            if (first == null) {
                first = theirs;
            } else {
                String complaint = first.mismatch(theirs);
                if (complaint != null) {
                    throw new IllegalArgumentException(member.path()
                            + " cannot be blended with " + bundle.members().get(0).path()
                            + ": " + complaint);
                }
            }
        }

        this.sessions = List.copyOf(opened);
        this.device = chosen;
        this.config = first.renamed(bundle.name());
        this.mixing = bundle.mixingMatrix(config.sourceCount());

        this.batch = options.batch > 0 ? options.batch
                : device == SessionFactory.Device.CPU ? 1 : 4;
        if (batch > 1 && !config.dynamicBatch()) {
            throw new IllegalArgumentException(
                    "this graph has a fixed batch of 1; re-export it with tools/export_onnx.py, "
                            + "which enables a dynamic batch axis");
        }

        this.spectrogram =
                new Spectrogram(config.nfft(), config.hop(), config.channels(), config.segment());
        if (spectrogram.frames() != config.frames()) {
            throw new IllegalArgumentException("the STFT produces " + spectrogram.frames()
                    + " frames but the graph expects " + config.frames());
        }

        if (mixing != null) {
            partSpec = new float[batch * config.sourceCount() * 2 * config.channels()
                    * config.freqBins() * config.frames()];
            partWav = new float[batch * config.sourceCount() * config.channels()
                    * config.segment()];
        }

        this.weight = OverlapAdd.transitionWeight(config.segment(), options.transitionPower);
        this.prefetch = options.pipeline
                ? Executors.newSingleThreadExecutor(runnable -> {
                    Thread thread = new Thread(runnable, "demucs-prefetch");
                    thread.setDaemon(true);
                    return thread;
                })
                : null;
    }

    public ModelConfig config() {
        return config;
    }

    public SessionFactory.Device device() {
        return device;
    }

    public int batch() {
        return batch;
    }

    /** How many networks each chunk goes through. */
    public int models() {
        return sessions.size();
    }

    /**
     * Separates planar {@code [channels][samples]} audio, handing finalised spans to {@code sink}.
     *
     * @param progress receives {@code (done, total)} chunk counts, or null
     */
    public void separate(float[] mix, int samples, StemSink sink, ProgressListener progress)
            throws OrtException, IOException {
        if (samples <= 0) {
            return;
        }

        int sources = config.sourceCount();
        int channels = config.channels();
        int segment = config.segment();
        float[] stats = referenceStats(mix, samples, channels);
        float mean = stats[0];
        float gain = stats[1];

        int stride = Math.max(1, (int) ((1.0f - options.overlap) * segment));
        int chunkCount = (samples + stride - 1) / stride;
        int groups = (chunkCount + batch - 1) / batch;

        Frame[] frames = {new Frame(batch, config), new Frame(batch, config)};
        OverlapAdd accumulator =
                new OverlapAdd(sources * channels, segment + (batch - 1) * stride);
        float[] stems = new float[batch * sources * channels * segment];
        float[] ready = new float[sources * channels * (segment + (batch - 1) * stride)];

        fillFrame(frames[0], 0, stride, mix, samples, mean, gain);

        for (int group = 0; group < groups; group++) {
            Frame current = frames[group % 2];
            Frame next = frames[(group + 1) % 2];

            // Build the next batch's spectrogram while this one is in the network. The two touch
            // disjoint buffers, and the Spectrogram itself is immutable.
            Future<?> pending = null;
            if (group + 1 < groups) {
                int nextGroup = group + 1;
                Runnable build = () -> fillFrame(next, nextGroup, stride, mix, samples, mean, gain);
                if (prefetch != null) {
                    pending = prefetch.submit(build);
                } else {
                    build.run();
                }
            }

            int count = current.count;
            infer(current, count, stems);

            for (int slot = 0; slot < count; slot++) {
                Chunk chunk = current.chunks[slot];
                int at = slot * sources * channels * segment + chunk.trim;
                accumulator.add(chunk.offset, stems, at, segment, chunk.length, weight);
            }

            long upto = Math.min(samples, current.chunks[count - 1].offset + stride);
            int flushed = accumulator.flush(upto, ready);
            if (flushed > 0) {
                int total = sources * channels * flushed;
                for (int i = 0; i < total; i++) {
                    ready[i] = ready[i] * gain + mean;
                }
                sink.accept(accumulator.base() - flushed, flushed, ready);
            }

            if (pending != null) {
                await(pending);
            }
            if (progress != null) {
                progress.update(Math.min(chunkCount, (group + 1) * batch), chunkCount);
            }
        }
    }

    /**
     * Convenience wrapper that keeps the whole result in memory, laid out as
     * {@code [sources][channels][samples]}.
     */
    public float[] separateToArray(float[] mix, int samples) throws OrtException, IOException {
        int planes = config.sourceCount() * config.channels();
        float[] output = new float[planes * samples];
        separate(mix, samples, (offset, count, block) -> {
            for (int plane = 0; plane < planes; plane++) {
                System.arraycopy(block, plane * count, output,
                        plane * samples + (int) offset, count);
            }
        }, null);
        return output;
    }

    /** Times the full pipeline on synthetic audio. */
    public Benchmark benchmark(double seconds) throws OrtException, IOException {
        int frames = (int) (seconds * config.sampleRate());
        java.util.Random random = new java.util.Random(0);
        float[] noise = new float[config.channels() * frames];
        for (int i = 0; i < noise.length; i++) {
            noise[i] = (float) (random.nextDouble() * 0.1 - 0.05);
        }

        // One warm-up chunk so the measured run is not paying for lazy kernel selection.
        separate(noise, config.segment(), (offset, count, block) -> { }, null);

        long started = System.nanoTime();
        separate(noise, frames, (offset, count, block) -> { }, null);
        double elapsed = (System.nanoTime() - started) / 1e9;

        return new Benchmark(device, batch, sessions.size(), seconds, elapsed);
    }

    @Override
    public void close() throws OrtException {
        if (prefetch != null) {
            prefetch.shutdown();
        }
        for (OrtSession open : sessions) {
            open.close();
        }
    }

    /**
     * demucs normalises by the mean and standard deviation of the mono downmix of the whole track.
     */
    private static float[] referenceStats(float[] mix, int samples, int channels) {
        double sum = 0.0;
        double sumSquares = 0.0;
        for (int i = 0; i < samples; i++) {
            double value = 0.0;
            for (int c = 0; c < channels; c++) {
                value += mix[c * samples + i];
            }
            value /= channels;
            sum += value;
            sumSquares += value * value;
        }
        double mean = sum / samples;
        double variance = Math.max(0.0, sumSquares / samples - mean * mean);
        return new float[] {(float) mean, (float) Math.sqrt(variance) + 1e-8f};
    }

    private void fillFrame(Frame frame, int group, int stride, float[] mix, int samples,
                           float mean, float gain) {
        int channels = config.channels();
        int segment = config.segment();

        frame.count = 0;
        Arrays.fill(frame.mix, 0.0f);

        for (int slot = 0; slot < batch; slot++) {
            long offset = (long) (group * batch + slot) * stride;
            if (offset >= samples) {
                break;
            }

            int length = (int) Math.min(samples - offset, segment);
            int delta = segment - length;
            // Short chunks are centred, exactly as demucs does it, so `start` may fall before the
            // track and `at` shifts the copy.
            long start = offset - delta / 2;
            long from = Math.max(0, start);
            long to = Math.min(samples, start + segment);
            int at = (int) (from - start);

            for (int c = 0; c < channels; c++) {
                int destination = (slot * channels + c) * segment + at;
                int source = (int) (c * (long) samples + from);
                for (int i = 0; i < to - from; i++) {
                    frame.mix[destination + i] = (mix[source + i] - mean) / gain;
                }
            }
            frame.chunks[frame.count++] = new Chunk(offset, length, delta / 2);
        }

        spectrogram.forward(frame.mix, frame.count, frame.mag);
    }

    /** One inference call plus the ISTFT of its spectral branch. */
    /** Runs every member over the batch, blends, and leaves the stems in {@code stems}. */
    private void infer(Frame frame, int count, float[] stems) throws OrtException {
        int channels = config.channels();
        int sources = config.sourceCount();
        int segment = config.segment();

        int mixCount = count * channels * segment;
        int magCount = count * 2 * channels * config.freqBins() * config.frames();
        long[] mixShape = {count, channels, segment};
        long[] magShape = {count, 2L * channels, config.freqBins(), config.frames()};

        try (OnnxTensor mixTensor = OnnxTensor.createTensor(
                     environment, FloatBuffer.wrap(frame.mix, 0, mixCount), mixShape);
             OnnxTensor magTensor = OnnxTensor.createTensor(
                     environment, FloatBuffer.wrap(frame.mag, 0, magCount), magShape)) {

            Map<String, OnnxTensor> inputs = new HashMap<>(4);
            inputs.put(config.inputNames()[0], mixTensor);
            inputs.put(config.inputNames()[1], magTensor);

            if (mixing == null) {
                try (OrtSession.Result result = sessions.get(0).run(inputs)) {
                    // Read the spectral branch straight out of ONNX Runtime's own buffer; the
                    // ISTFT does random access anyway, so copying it first gains nothing.
                    FloatBuffer spec = ((OnnxTensor) result.get(0)).getFloatBuffer();
                    FloatBuffer wav = ((OnnxTensor) result.get(1)).getFloatBuffer();
                    checkFinite(spec);
                    checkSize(spec.remaining(), magCount * sources, sources);
                    spectrogram.inverse(spec, count * sources, stems);
                    addWaveformBranch(stems, wav, count * sources * channels * segment);
                }
                return;
            }

            int specPlane = 2 * channels * config.freqBins() * config.frames();
            int wavPlane = channels * segment;
            boolean written = false;
            for (int model = 0; model < sessions.size(); model++) {
                float[] weights = mixing[model];
                if (allZero(weights)) {
                    continue;                    // this member owns none of the stems
                }
                try (OrtSession.Result result = sessions.get(model).run(inputs)) {
                    FloatBuffer spec = ((OnnxTensor) result.get(0)).getFloatBuffer();
                    FloatBuffer wav = ((OnnxTensor) result.get(1)).getFloatBuffer();
                    checkFinite(spec);
                    checkSize(spec.remaining(), magCount * sources, sources);
                    blend(partSpec, spec, weights, count, specPlane, !written);
                    blend(partWav, wav, weights, count, wavPlane, !written);
                    written = true;
                }
            }

            spectrogram.inverse(FloatBuffer.wrap(partSpec, 0, count * sources * specPlane),
                    count * sources, stems);
            int used = count * sources * wavPlane;
            for (int i = 0; i < used; i++) {
                stems[i] += partWav[i];
            }
        }
    }

    /** {@code out[b][s] = (+)= weight[s] * part[b][s]}, over {@code plane} values per stem. */
    private static void blend(float[] out, FloatBuffer part, float[] weights, int count,
                              int plane, boolean overwrite) {
        int base = part.position();
        for (int slot = 0; slot < count * weights.length; slot++) {
            float weight = weights[slot % weights.length];
            int at = slot * plane;
            if (overwrite) {
                for (int i = 0; i < plane; i++) {
                    out[at + i] = weight * part.get(base + at + i);
                }
            } else if (weight != 0.0f) {
                for (int i = 0; i < plane; i++) {
                    out[at + i] += weight * part.get(base + at + i);
                }
            }
        }
    }

    private static boolean allZero(float[] weights) {
        for (float weight : weights) {
            if (weight != 0.0f) {
                return false;
            }
        }
        return true;
    }

    /**
     * A provider that overflows -- TensorRT asked for fp16 on this model is the known case --
     * returns NaN for every sample, and the only symptom downstream is a file full of silence.
     * Checked on the first batch only.
     */
    private void checkFinite(FloatBuffer spec) {
        if (checkedFinite) {
            return;
        }
        checkedFinite = true;
        for (int i = spec.position(); i < spec.limit(); i++) {
            if (!Float.isFinite(spec.get(i))) {
                throw new IllegalStateException("the network returned a non-finite '"
                        + config.outputNames()[0] + "'. Half precision overflows on this model "
                        + "with some providers; use the fp32 graph, or the fp16 one from "
                        + "export_onnx.py --fp16 rather than asking the provider to convert");
            }
        }
    }

    /**
     * The graph declares its outputs symbolically, so this is the first point at which the stem
     * count can be checked against what the network produced.
     */
    private void checkSize(int got, int expected, int sources) {
        if (got != expected) {
            throw new IllegalStateException("'" + config.outputNames()[0] + "' holds " + got
                    + " values, expected " + expected + " for " + sources
                    + " stems; the sidecar and the graph disagree");
        }
    }

    /**
     * The network splits its prediction between a spectral and a waveform branch; the stem is the
     * sum of the two.
     */
    private static void addWaveformBranch(float[] stems, FloatBuffer wav, int used) {
        int base = wav.position();
        for (int i = 0; i < used; i++) {
            stems[i] += wav.get(base + i);
        }
    }

    private static void await(Future<?> pending) {
        try {
            pending.get();
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted while preparing the next batch", error);
        } catch (java.util.concurrent.ExecutionException error) {
            throw new IllegalStateException(error.getCause());
        }
    }

    /** Receives chunk progress. */
    @FunctionalInterface
    public interface ProgressListener {
        void update(int done, int total);
    }

    /** One chunk's place in the track. */
    private record Chunk(long offset, int length, int trim) {
    }

    /** One batch's inputs, filled ahead of the call that consumes them. */
    private static final class Frame {
        final float[] mix;
        final float[] mag;
        final Chunk[] chunks;
        int count;

        Frame(int batch, ModelConfig config) {
            mix = new float[batch * config.channels() * config.segment()];
            mag = new float[batch * 2 * config.channels() * config.freqBins() * config.frames()];
            chunks = new Chunk[batch];
        }
    }

    /** Convenience: the stem names this separator will produce. */
    public List<String> sources() {
        return config.sources();
    }
}
