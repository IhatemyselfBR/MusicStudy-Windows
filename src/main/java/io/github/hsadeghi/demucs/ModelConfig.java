// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Model geometry, resolved from the ONNX graph itself.
 *
 * <p>Tensor shapes come from the graph, which cannot disagree with itself; the human-readable parts
 * -- stem names, sample rate -- come from the sidecar JSON written by {@code tools/export_onnx.py}.
 */
public final class ModelConfig {

    private static final String[] DEFAULT_SOURCES = {"drums", "bass", "other", "vocals"};

    private final String name;
    private final List<String> sources;
    private final int sampleRate;
    private final int channels;
    private final int segment;
    private final int nfft;
    private final int hop;
    private final int freqBins;
    private final int frames;
    private final boolean dynamicBatch;
    private final String[] inputNames;
    private final String[] outputNames;

    private ModelConfig(String name, List<String> sources, int sampleRate, int channels,
                        int segment, int nfft, int hop, int freqBins, int frames,
                        boolean dynamicBatch, String[] inputNames, String[] outputNames) {
        this.name = name;
        this.sources = List.copyOf(sources);
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.segment = segment;
        this.nfft = nfft;
        this.hop = hop;
        this.freqBins = freqBins;
        this.frames = frames;
        this.dynamicBatch = dynamicBatch;
        this.inputNames = inputNames;
        this.outputNames = outputNames;
    }

    public String name() {
        return name;
    }

    public List<String> sources() {
        return sources;
    }

    public int sourceCount() {
        return sources.size();
    }

    public int sampleRate() {
        return sampleRate;
    }

    public int channels() {
        return channels;
    }

    /** Samples per chunk fed to the network. */
    public int segment() {
        return segment;
    }

    public int nfft() {
        return nfft;
    }

    public int hop() {
        return hop;
    }

    /** {@code nfft / 2}; the Nyquist bin is dropped. */
    public int freqBins() {
        return freqBins;
    }

    public int frames() {
        return frames;
    }

    /** Whether the graph accepts more than one chunk per call. */
    public boolean dynamicBatch() {
        return dynamicBatch;
    }

    public String[] inputNames() {
        return inputNames.clone();
    }

    public String[] outputNames() {
        return outputNames.clone();
    }

    public double segmentSeconds() {
        return (double) segment / sampleRate;
    }

    /** Reads the geometry of an open session plus its sidecar file. */
    public static ModelConfig fromSession(OrtSession session, Path modelPath) throws OrtException {
        String[] inputNames = session.getInputNames().toArray(String[]::new);
        String[] outputNames = session.getOutputNames().toArray(String[]::new);
        if (inputNames.length != 2 || outputNames.length != 2) {
            throw new IllegalArgumentException("expected a 2-in/2-out graph, got "
                    + inputNames.length + "/" + outputNames.length);
        }

        Map<String, NodeInfo> inputs = session.getInputInfo();
        Map<String, NodeInfo> outputs = session.getOutputInfo();
        long[] mix = shapeOf(inputs.get(inputNames[0]));
        long[] mag = shapeOf(inputs.get(inputNames[1]));
        long[] spec = shapeOf(outputs.get(outputNames[0]));
        if (mix.length != 3 || mag.length != 4 || spec.length != 5) {
            throw new IllegalArgumentException("unexpected tensor ranks: mix " + mix.length
                    + ", mag " + mag.length + ", spec " + spec.length);
        }

        Sidecar sidecar = Sidecar.beside(modelPath);
        int channels = dimension(mix[1], sidecar.integer("audio_channels"), "mix channels");
        int segment = dimension(mix[2], sidecar.integer("segment_samples"), "mix length");
        int freqBins = dimension(mag[2], sidecar.integer("freq_bins"), "frequency bins");
        int frames = dimension(mag[3], sidecar.integer("frames"), "frames");

        // torch.onnx.export cannot prove the output shapes, so it leaves every one of their
        // dimensions symbolic, which arrives here as -1. The stem count therefore comes from the
        // sidecar, and the graph is only consulted when it does state a number.
        int declared = spec[1] > 0 ? (int) spec[1] : -1;
        int wanted = declared > 0 ? declared : DEFAULT_SOURCES.length;
        List<String> sources = sidecar.strings("sources").orElseGet(() -> {
            List<String> defaults = new ArrayList<>(wanted);
            for (int i = 0; i < wanted; i++) {
                defaults.add(i < DEFAULT_SOURCES.length ? DEFAULT_SOURCES[i] : "stem" + i);
            }
            return defaults;
        });
        if (declared > 0 && sources.size() != declared) {
            throw new IllegalArgumentException("the sidecar lists " + sources.size()
                    + " stems but the graph produces " + declared);
        }

        return new ModelConfig(
                sidecar.string("name").orElse(Sidecar.stem(modelPath)),
                sources,
                sidecar.integer("samplerate").orElse(44100),
                channels,
                segment,
                // every demucs model uses 75 % frame overlap unless the sidecar says so.
                sidecar.integer("nfft").orElse(freqBins * 2),
                sidecar.integer("hop").orElse(freqBins / 2),
                freqBins,
                frames,
                mix[0] < 0,
                inputNames,
                outputNames);
    }

    /** Renames a configuration; a bag takes its name from the manifest, not from member zero. */
    ModelConfig renamed(String newName) {
        return new ModelConfig(newName, sources, sampleRate, channels, segment, nfft, hop,
                freqBins, frames, dynamicBatch, inputNames, outputNames);
    }

    /** Why two graphs cannot be blended, or {@code null} if they can. */
    String mismatch(ModelConfig other) {
        if (!sources.equals(other.sources)) {
            return "they separate different stems";
        }
        if (sampleRate != other.sampleRate || channels != other.channels) {
            return "they take different audio";
        }
        if (segment != other.segment || frames != other.frames || nfft != other.nfft) {
            return "they take differently shaped tensors";
        }
        return null;
    }

    @Override
    public String toString() {
        return String.format("%s: %d stems [%s], %d Hz, %d ch, segment %d samples (%.2f s), "
                        + "nfft %d/hop %d -> %dx%d, batch %s",
                name, sourceCount(), String.join(", ", sources), sampleRate, channels, segment,
                segmentSeconds(), nfft, hop, freqBins, frames,
                dynamicBatch ? "dynamic" : "fixed at 1");
    }

    /**
     * A concrete dimension from the graph, or the sidecar's word for it if the graph left it
     * symbolic, which ONNX Runtime reports as -1.
     */
    private static int dimension(long value, Optional<Integer> fallback, String label) {
        if (value > 0) {
            return (int) value;
        }
        return fallback.orElseThrow(() -> new IllegalArgumentException(
                "the graph leaves " + label + " symbolic and the sidecar does not say"));
    }

    private static long[] shapeOf(NodeInfo node) {
        if (node == null || !(node.getInfo() instanceof TensorInfo tensor)) {
            throw new IllegalArgumentException("expected a tensor input/output");
        }
        return tensor.getShape();
    }
}
