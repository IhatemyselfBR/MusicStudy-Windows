// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs.audio;

/**
 * Planar float32 audio: {@code samples} holds channel 0 in full, then channel 1, and so on. That is
 * the layout the model and the spectrogram want, so nothing downstream has to de-interleave.
 */
public final class AudioBuffer {

    private float[] samples;
    private final int channels;
    private int frames;
    private final int sampleRate;

    public AudioBuffer(int channels, int frames, int sampleRate) {
        this.channels = channels;
        this.frames = frames;
        this.sampleRate = sampleRate;
        this.samples = new float[channels * frames];
    }

    public float[] samples() {
        return samples;
    }

    public int channels() {
        return channels;
    }

    public int frames() {
        return frames;
    }

    public int sampleRate() {
        return sampleRate;
    }

    public double duration() {
        return sampleRate == 0 ? 0.0 : (double) frames / sampleRate;
    }

    /** Index of the first sample of channel {@code index}. */
    public int channelOffset(int index) {
        return index * frames;
    }

    /** Keeps only the first {@code count} samples of every channel. */
    public void truncate(int count) {
        if (count >= frames) {
            return;
        }
        float[] trimmed = new float[channels * count];
        for (int c = 0; c < channels; c++) {
            System.arraycopy(samples, c * frames, trimmed, c * count, count);
        }
        samples = trimmed;
        frames = count;
    }

    /** Folds or duplicates channels to reach exactly {@code target}. */
    public AudioBuffer remix(int target) {
        if (channels == target) {
            return this;
        }

        AudioBuffer output = new AudioBuffer(target, frames, sampleRate);
        if (channels == 1) {
            for (int c = 0; c < target; c++) {
                System.arraycopy(samples, 0, output.samples, c * frames, frames);
            }
        } else if (target == 1) {
            float scale = 1.0f / channels;
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < frames; i++) {
                    output.samples[i] += samples[c * frames + i] * scale;
                }
            }
        } else {
            for (int c = 0; c < target; c++) {
                int source = Math.min(c, channels - 1);
                System.arraycopy(samples, source * frames, output.samples, c * frames, frames);
            }
        }
        return output;
    }
}
