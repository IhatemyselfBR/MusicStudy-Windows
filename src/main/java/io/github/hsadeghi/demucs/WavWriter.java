// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs.audio;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Incremental WAV writer. The separator finalises its output front-to-back, so stems can be written
 * as they are produced and never held whole in memory.
 */
public final class WavWriter implements AutoCloseable {

    /** Output sample format for the stem files. */
    public enum Format {
        FLOAT32(4, 3),
        INT24(3, 1),
        INT16(2, 1);

        final int width;
        final int tag;

        Format(int width, int tag) {
            this.width = width;
            this.tag = tag;
        }

        public static Format parse(String text) {
            return switch (text) {
                case "float" -> FLOAT32;
                case "int24" -> INT24;
                case "int16" -> INT16;
                default -> throw new IllegalArgumentException("unknown format '" + text + "'");
            };
        }
    }

    private final Path path;
    private final OutputStream stream;
    private final int channels;
    private final Format format;
    private long frames;
    private byte[] scratch = new byte[0];

    public WavWriter(Path path, int sampleRate, int channels, Format format) throws IOException {
        this.path = path;
        this.channels = channels;
        this.format = format;
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        this.stream = new BufferedOutputStream(Files.newOutputStream(path), 1 << 16);
        writeHeader(sampleRate);
    }

    public Path path() {
        return path;
    }

    /**
     * Writes {@code channels} rows of {@code count} samples, {@code stride} apart, starting at
     * {@code offset} in {@code planar}.
     */
    public void write(float[] planar, int offset, int stride, int count) throws IOException {
        int bytes = count * channels * format.width;
        if (scratch.length < bytes) {
            scratch = new byte[bytes];
        }
        ByteBuffer buffer = ByteBuffer.wrap(scratch).order(ByteOrder.LITTLE_ENDIAN);

        for (int frame = 0; frame < count; frame++) {
            for (int c = 0; c < channels; c++) {
                encode(buffer, planar[offset + c * stride + frame]);
            }
        }
        stream.write(scratch, 0, bytes);
        frames += count;
    }

    /** Patches the two length fields, which are only known at the end. */
    @Override
    public void close() throws IOException {
        stream.close();
        long dataBytes = frames * channels * format.width;
        try (RandomAccessFile file = new RandomAccessFile(path.toFile(), "rw")) {
            file.seek(4);
            writeInt(file, (int) (36 + dataBytes));
            file.seek(40);
            writeInt(file, (int) dataBytes);
        }
    }

    private void encode(ByteBuffer buffer, float value) {
        switch (format) {
            case FLOAT32 -> buffer.putFloat(value);
            case INT24 -> {
                int sample = (int) (clamp(value) * 8388607.0f);
                buffer.put((byte) sample);
                buffer.put((byte) (sample >> 8));
                buffer.put((byte) (sample >> 16));
            }
            case INT16 -> buffer.putShort((short) (clamp(value) * 32767.0f));
        }
    }

    private static float clamp(float value) {
        return Math.max(-1.0f, Math.min(1.0f, value));
    }

    private void writeHeader(int sampleRate) throws IOException {
        ByteBuffer header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);
        header.put("RIFF".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        header.putInt(0);                                      // patched by close()
        header.put("WAVEfmt ".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        header.putInt(16);
        header.putShort((short) format.tag);
        header.putShort((short) channels);
        header.putInt(sampleRate);
        header.putInt(sampleRate * channels * format.width);
        header.putShort((short) (channels * format.width));
        header.putShort((short) (format.width * 8));
        header.put("data".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        header.putInt(0);                                      // patched by close()
        stream.write(header.array());
    }

    private static void writeInt(RandomAccessFile file, int value) throws IOException {
        file.write(new byte[] {
            (byte) value, (byte) (value >> 8), (byte) (value >> 16), (byte) (value >> 24),
        });
    }
}
