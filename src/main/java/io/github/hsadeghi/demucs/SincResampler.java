// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs.audio;

/**
 * Band-limited resampling by windowed-sinc interpolation.
 *
 * <p>htdemucs only ever runs at 44.1 kHz, so this exists to get other files there. The filter is a
 * Blackman-windowed sinc, tabulated once and read with linear interpolation between entries -- fine
 * enough that the interpolation error sits well below the quantisation floor, without a
 * transcendental per tap.
 */
public final class SincResampler {

    private final double ratio;
    private final double cutoff;
    private final double support;
    private final int half;
    private final int steps;
    private final float[] table;

    /**
     * @param ratio     output rate divided by input rate
     * @param halfWidth zero crossings on each side of the filter; 16 puts the stopband well below
     *                  16-bit noise for music material
     * @param steps     table entries per zero crossing
     */
    public SincResampler(double ratio, int halfWidth, int steps) {
        this.ratio = ratio;
        this.half = halfWidth;
        this.steps = steps;
        this.cutoff = Math.min(1.0, ratio);
        this.support = halfWidth / cutoff;

        this.table = new float[2 * halfWidth * steps + 1];
        for (int i = 0; i < table.length; i++) {
            double x = (double) i / steps - halfWidth;
            table[i] = (float) (sinc(x) * blackman(x / halfWidth));
        }
    }

    public int outputFrames(int inputFrames) {
        return (int) Math.round(inputFrames * ratio);
    }

    public void process(float[] input, int inputOffset, int inputFrames, float[] output,
                        int outputOffset, int outputFrames) {
        for (int i = 0; i < outputFrames; i++) {
            double centre = i / ratio;
            long first = (long) Math.ceil(centre - support);
            long last = (long) Math.floor(centre + support);

            double sum = 0.0;
            for (long j = first; j <= last; j++) {
                int clamped = (int) Math.max(0, Math.min(j, inputFrames - 1));
                sum += input[inputOffset + clamped] * tap((centre - j) * cutoff);
            }
            output[outputOffset + i] = (float) (sum * cutoff);
        }
    }

    public static AudioBuffer resample(AudioBuffer input, int targetRate) {
        if (input.sampleRate() == targetRate || input.frames() == 0) {
            return input;
        }

        SincResampler resampler =
                new SincResampler((double) targetRate / input.sampleRate(), 16, 512);
        int frames = resampler.outputFrames(input.frames());
        AudioBuffer output = new AudioBuffer(input.channels(), frames, targetRate);

        for (int c = 0; c < input.channels(); c++) {
            resampler.process(input.samples(), input.channelOffset(c), input.frames(),
                    output.samples(), output.channelOffset(c), frames);
        }
        return output;
    }

    private double tap(double x) {
        double position = (x + half) * steps;
        if (position <= 0.0 || position >= table.length - 1) {
            return 0.0;
        }
        int index = (int) position;
        double fraction = position - index;
        return table[index] * (1.0 - fraction) + table[index + 1] * fraction;
    }

    private static double sinc(double x) {
        if (Math.abs(x) < 1e-9) {
            return 1.0;
        }
        double t = Math.PI * x;
        return Math.sin(t) / t;
    }

    private static double blackman(double t) {
        if (Math.abs(t) >= 1.0) {
            return 0.0;
        }
        double phase = Math.PI * (t + 1.0);
        return 0.42 - 0.5 * Math.cos(phase) + 0.08 * Math.cos(2.0 * phase);
    }
}
