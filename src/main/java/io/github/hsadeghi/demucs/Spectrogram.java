// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs;

import java.nio.FloatBuffer;
import java.util.stream.IntStream;

/**
 * The complex half of HTDemucs, which ONNX cannot express.
 *
 * <p>{@code torch.stft} / {@code torch.istft} have no ONNX equivalent -- there is no ISTFT operator
 * at all -- so the exported graph is cut at the complex boundary and the spectrogram lives here.
 * This reproduces {@code HTDemucs._spec} and {@code HTDemucs._ispec}, padding quirks included:
 * reflect-pad by 3*hop/2, STFT with a periodic Hann window (normalised, centred), drop the Nyquist
 * bin, drop two frames at each end -- and the exact inverse of all that.
 *
 * <p>Spectrograms use the "channels as complex" layout the model was trained with: plane 2c is the
 * real part of channel c, plane 2c+1 the imaginary part.
 *
 * <p>Instances are immutable after construction and safe to use from several threads at once, which
 * is what lets the separator overlap a forward transform with an inverse one.
 */
public final class Spectrogram {

    private final int nfft;
    private final int hop;
    private final int channels;
    private final int length;
    private final int freqBins;
    private final int frames;
    private final int pad;
    private final int rightPad;
    private final int padded;
    private final int centred;
    private final int totalSamples;
    private final int trim;
    private final float forwardScale;
    private final float inverseScale;
    private final float[] window;
    private final float[] envelope;
    private final RealFft fft;

    public Spectrogram(int nfft, int hop, int channels, int length) {
        if (nfft % hop != 0) {
            throw new IllegalArgumentException("nfft must be a multiple of hop");
        }
        this.nfft = nfft;
        this.hop = hop;
        this.channels = channels;
        this.length = length;

        this.freqBins = nfft / 2;                              // Nyquist bin dropped
        this.frames = (length + hop - 1) / hop;                // demucs' `le`
        this.pad = hop / 2 * 3;
        this.rightPad = pad + frames * hop - length;
        this.padded = length + pad + rightPad;
        this.centred = padded + nfft;
        this.trim = nfft / 2 + pad;
        this.forwardScale = (float) (1.0 / Math.sqrt(nfft));
        this.inverseScale = (float) Math.sqrt(nfft);
        this.fft = new RealFft(nfft);

        int totalFrames = 1 + padded / hop;                    // == frames + 4
        this.totalSamples = nfft + hop * (totalFrames - 1);

        this.window = hannPeriodic(nfft);
        this.envelope = new float[totalSamples];
        for (int t = 0; t < totalFrames; t++) {
            for (int j = 0; j < nfft; j++) {
                envelope[t * hop + j] += window[j] * window[j];
            }
        }
    }

    /** Frequency bins kept, {@code nfft / 2}. */
    public int freqBins() {
        return freqBins;
    }

    /** STFT frames produced for one chunk. */
    public int frames() {
        return frames;
    }

    /** {@code torch.hann_window(n)} -- periodic, not the symmetric variant. */
    public static float[] hannPeriodic(int n) {
        float[] window = new float[n];
        for (int i = 0; i < n; i++) {
            window[i] = (float) (0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / n)));
        }
        return window;
    }

    /** numpy/torch {@code reflect} padding: {@code out[-1] == in[1]}, {@code out[len] == in[len-2]}. */
    public static void reflectPad(float[] input, int from, int n, int left, int right,
                                  float[] output, int at) {
        for (int i = 0; i < left; i++) {
            output[at + i] = input[from + left - i];
        }
        System.arraycopy(input, from, output, at + left, n);
        for (int i = 0; i < right; i++) {
            output[at + left + n + i] = input[from + n - 2 - i];
        }
    }

    /** {@code [batch][channels][length]} -> {@code [batch][2*channels][freqBins][frames]}. */
    public void forward(float[] mix, int batch, float[] output) {
        int planes = batch * channels;
        float[] padding = new float[planes * centred];

        IntStream.range(0, planes).parallel().forEach(plane -> {
            float[] inner = new float[padded];
            reflectPad(mix, plane * length, length, pad, rightPad, inner, 0);
            reflectPad(inner, 0, padded, nfft / 2, nfft / 2, padding, plane * centred);
        });

        int planeSize = freqBins * frames;
        IntStream.range(0, planes * frames).parallel().forEach(index -> {
            Scratch scratch = Scratch.forThread(this);
            int plane = index / frames;
            int frame = index % frames;
            int item = plane / channels;
            int channel = plane % channels;

            int at = plane * centred + (frame + 2) * hop;
            for (int j = 0; j < nfft; j++) {
                scratch.windowed[j] = padding[at + j] * window[j];
            }
            fft.forward(scratch.windowed, scratch.re, scratch.im, scratch.fft);

            int reBase = (item * 2 * channels + 2 * channel) * planeSize + frame;
            int imBase = reBase + planeSize;
            for (int f = 0; f < freqBins; f++) {
                output[reBase + f * frames] = scratch.re[f] * forwardScale;
                output[imBase + f * frames] = scratch.im[f] * forwardScale;
            }
        });
    }

    /** {@code [items][2*channels][freqBins][frames]} -> {@code [items][channels][length]}. */
    public void inverse(float[] spec, int items, float[] output) {
        inverse(FloatBuffer.wrap(spec), items, output);
    }

    /**
     * Same, reading the spectrogram straight out of an ONNX Runtime output buffer so the network's
     * result never has to be copied into a Java array first.
     */
    public void inverse(FloatBuffer spec, int items, float[] output) {
        int planes = items * channels;
        int planeSize = freqBins * frames;

        IntStream.range(0, planes).parallel().forEach(plane -> {
            Scratch scratch = Scratch.forThread(this);
            java.util.Arrays.fill(scratch.accumulator, 0.0f);

            int item = plane / channels;
            int channel = plane % channels;
            int reBase = (item * 2 * channels + 2 * channel) * planeSize;
            int imBase = reBase + planeSize;

            // The dropped Nyquist bin stays zero for every frame.
            scratch.re[freqBins] = 0.0f;
            scratch.im[freqBins] = 0.0f;

            for (int frame = 0; frame < frames; frame++) {
                for (int f = 0; f < freqBins; f++) {
                    scratch.re[f] = spec.get(reBase + f * frames + frame);
                    scratch.im[f] = spec.get(imBase + f * frames + frame);
                }
                fft.inverse(scratch.re, scratch.im, scratch.windowed, scratch.fft);

                int at = (frame + 2) * hop;                    // two empty frames were prepended
                for (int j = 0; j < nfft; j++) {
                    scratch.accumulator[at + j] += scratch.windowed[j] * inverseScale * window[j];
                }
            }

            int outBase = plane * length;
            for (int i = 0; i < length; i++) {
                output[outBase + i] = scratch.accumulator[trim + i] / envelope[trim + i];
            }
        });
    }

    /**
     * Per-thread working buffers. The parallel streams above run a few hundred short tasks per
     * chunk, so allocating inside each one would dominate; a thread-local set is reused instead.
     */
    private static final class Scratch {
        private static final ThreadLocal<Scratch> CACHE = new ThreadLocal<>();

        final float[] windowed;
        final float[] re;
        final float[] im;
        final float[] fft;
        final float[] accumulator;

        Scratch(Spectrogram owner) {
            windowed = new float[owner.nfft];
            re = new float[owner.freqBins + 1];
            im = new float[owner.freqBins + 1];
            fft = new float[owner.fft.scratchLength()];
            accumulator = new float[owner.totalSamples];
        }

        static Scratch forThread(Spectrogram owner) {
            Scratch scratch = CACHE.get();
            if (scratch == null || scratch.windowed.length != owner.nfft
                    || scratch.accumulator.length != owner.totalSamples) {
                scratch = new Scratch(owner);
                CACHE.set(scratch);
            }
            return scratch;
        }
    }
}
