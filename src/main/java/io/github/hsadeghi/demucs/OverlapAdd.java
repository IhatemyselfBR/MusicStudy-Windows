// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs;

import java.util.Arrays;

/**
 * Weighted overlap-add over a window that slides with the input.
 *
 * <p>Chunks are added at absolute sample offsets; once a sample can no longer be touched by a
 * future chunk it is normalised by the accumulated weight and handed back. Only {@code capacity}
 * samples are ever held, so memory does not grow with track length.
 */
public final class OverlapAdd {

    private final int planes;
    private final int capacity;
    private final float[] accumulator;
    private final float[] weight;
    private long base;

    public OverlapAdd(int planes, int capacity) {
        this.planes = planes;
        this.capacity = capacity;
        this.accumulator = new float[planes * capacity];
        this.weight = new float[capacity];
    }

    /** Samples held at once. */
    public int capacity() {
        return capacity;
    }

    /** Absolute index of the first sample still held. */
    public long base() {
        return base;
    }

    /**
     * Adds one row per plane of {@code count} samples, {@code stride} apart in {@code block}
     * starting at {@code blockOffset}, at absolute sample {@code offset}, under the first
     * {@code count} values of {@code weights}.
     */
    public void add(long offset, float[] block, int blockOffset, int stride, int count,
                    float[] weights) {
        long start = offset - base;
        if (start < 0 || start + count > capacity) {
            throw new IndexOutOfBoundsException("chunk does not fit the overlap-add window");
        }

        for (int plane = 0; plane < planes; plane++) {
            int source = blockOffset + plane * stride;
            int destination = (int) (plane * (long) capacity + start);
            for (int i = 0; i < count; i++) {
                accumulator[destination + i] += block[source + i] * weights[i];
            }
        }
        for (int i = 0; i < count; i++) {
            weight[(int) start + i] += weights[i];
        }
    }

    /**
     * Finalises everything before absolute sample {@code upto} into {@code output}, laid out as one
     * row per plane of the returned length. Returns 0 if nothing is ready.
     *
     * @param output a buffer at least {@code planes * (upto - base())} long
     */
    public int flush(long upto, float[] output) {
        if (upto <= base) {
            return 0;
        }
        int count = (int) (upto - base);

        for (int plane = 0; plane < planes; plane++) {
            int source = plane * capacity;
            int destination = plane * count;
            for (int i = 0; i < count; i++) {
                output[destination + i] = accumulator[source + i] / weight[i];
            }
        }

        int keep = capacity - count;
        for (int plane = 0; plane < planes; plane++) {
            int row = plane * capacity;
            System.arraycopy(accumulator, row + count, accumulator, row, keep);
            Arrays.fill(accumulator, row + keep, row + capacity, 0.0f);
        }
        System.arraycopy(weight, count, weight, 0, keep);
        Arrays.fill(weight, keep, capacity, 0.0f);

        base = upto;
        return count;
    }

    /**
     * demucs' triangular cross-fade: up over the first half, down over the second. Chunk boundaries
     * then land exactly where upstream puts them.
     */
    public static float[] transitionWeight(int segment, float power) {
        float[] weight = new float[segment];
        float peak = 0.0f;
        for (int i = 0; i < segment; i++) {
            weight[i] = i < segment / 2 ? i + 1 : segment - i;
            peak = Math.max(peak, weight[i]);
        }
        for (int i = 0; i < segment; i++) {
            weight[i] /= peak;
            if (power != 1.0f) {
                weight[i] = (float) Math.pow(weight[i], power);
            }
        }
        return weight;
    }
}
