// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs;

/**
 * Radix-2 real FFT.
 *
 * <p>Conventions match {@code numpy.fft.rfft} / {@code numpy.fft.irfft}: the forward transform is
 * unnormalised and the inverse carries the {@code 1/n}. A length-n real transform is computed as a
 * length-n/2 complex transform plus a recombination step, which is the usual factor-of-two saving.
 *
 * <p>Instances hold nothing but immutable twiddle tables, so one can be shared by any number of
 * threads; per-call scratch is supplied by the caller.
 */
public final class RealFft {

    private final int size;
    private final int half;
    private final float[] cos;
    private final float[] sin;
    private final float[] cosN;
    private final float[] sinN;
    private final int[] reverse;

    /** @param size a power of two, at least 4 */
    public RealFft(int size) {
        if (size < 4 || (size & (size - 1)) != 0) {
            throw new IllegalArgumentException("FFT size must be a power of two >= 4");
        }
        this.size = size;
        this.half = size / 2;

        cos = new float[half / 2];
        sin = new float[half / 2];
        for (int i = 0; i < half / 2; i++) {
            double angle = -2.0 * Math.PI * i / half;
            cos[i] = (float) Math.cos(angle);
            sin[i] = (float) Math.sin(angle);
        }

        cosN = new float[half];
        sinN = new float[half];
        for (int k = 0; k < half; k++) {
            double angle = -2.0 * Math.PI * k / size;
            cosN[k] = (float) Math.cos(angle);
            sinN[k] = (float) Math.sin(angle);
        }

        int bits = Integer.numberOfTrailingZeros(half);
        reverse = new int[half];
        for (int i = 0; i < half; i++) {
            reverse[i] = Integer.reverse(i) >>> (32 - bits);
        }
    }

    public int size() {
        return size;
    }

    /** Number of output bins, {@code n/2 + 1}. */
    public int bins() {
        return half + 1;
    }

    /** Floats the caller must provide as scratch. */
    public int scratchLength() {
        return half * 2;
    }

    /** Transforms {@code input} (n real samples) into n/2+1 bins. */
    public void forward(float[] input, float[] re, float[] im, float[] scratch) {
        for (int k = 0; k < half; k++) {
            scratch[k] = input[2 * k];
            scratch[half + k] = input[2 * k + 1];
        }
        transform(scratch, false);

        int mask = half - 1;
        for (int k = 0; k < half; k++) {
            int j = (half - k) & mask;
            float ar = scratch[k], ai = scratch[half + k];
            float br = scratch[j], bi = -scratch[half + j];   // conj(Z[m - k])
            float evenR = 0.5f * (ar + br), evenI = 0.5f * (ai + bi);
            float oddR = 0.5f * (ai - bi), oddI = -0.5f * (ar - br);
            float wr = cosN[k], wi = sinN[k];
            re[k] = evenR + (oddR * wr - oddI * wi);
            im[k] = evenI + (oddR * wi + oddI * wr);
        }
        re[half] = scratch[0] - scratch[half];
        im[half] = 0.0f;
    }

    /** Transforms n/2+1 bins back into n real samples, {@code 1/n} included. */
    public void inverse(float[] re, float[] im, float[] output, float[] scratch) {
        for (int k = 0; k < half; k++) {
            // DC and Nyquist are taken as purely real, as numpy and torch do. A predicted
            // (non-physical) spectrum can carry a spurious imaginary part there, and keeping it
            // injects broadband noise.
            float xr = re[k];
            float xi = k == 0 ? 0.0f : im[k];
            float yr = re[half - k];
            float yi = k == 0 ? 0.0f : -im[half - k];
            float evenR = 0.5f * (xr + yr), evenI = 0.5f * (xi + yi);
            float diffR = 0.5f * (xr - yr), diffI = 0.5f * (xi - yi);
            float wr = cosN[k], wi = -sinN[k];               // e^{+2 pi i k / n}
            float oddR = diffR * wr - diffI * wi;
            float oddI = diffR * wi + diffI * wr;
            scratch[k] = evenR - oddI;
            scratch[half + k] = evenI + oddR;
        }
        transform(scratch, true);

        float scale = 1.0f / half;
        for (int k = 0; k < half; k++) {
            output[2 * k] = scratch[k] * scale;
            output[2 * k + 1] = scratch[half + k] * scale;
        }
    }

    /**
     * In-place iterative radix-2 decimation-in-time complex FFT, unscaled. {@code work} holds the
     * real parts in {@code [0, half)} and the imaginary parts in {@code [half, 2*half)}.
     */
    private void transform(float[] work, boolean inverse) {
        for (int i = 0; i < half; i++) {
            int j = reverse[i];
            if (j > i) {
                float tr = work[i];
                work[i] = work[j];
                work[j] = tr;
                float ti = work[half + i];
                work[half + i] = work[half + j];
                work[half + j] = ti;
            }
        }

        for (int len = 2; len <= half; len <<= 1) {
            int span = len >> 1;
            int step = half / len;
            for (int start = 0; start < half; start += len) {
                for (int k = 0, t = 0; k < span; k++, t += step) {
                    float wr = cos[t];
                    float wi = inverse ? -sin[t] : sin[t];
                    int a = start + k, b = a + span;
                    float ur = work[a], ui = work[half + a];
                    float vr = work[b] * wr - work[half + b] * wi;
                    float vi = work[b] * wi + work[half + b] * wr;
                    work[a] = ur + vr;
                    work[half + a] = ui + vi;
                    work[b] = ur - vr;
                    work[half + b] = ui - vi;
                }
            }
        }
    }
}
