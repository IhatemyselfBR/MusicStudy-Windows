// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs.audio;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Self-contained RIFF/WAVE reading: PCM 8/16/24/32 and IEEE float 32/64.
 *
 * <p>Other container formats are out of scope on purpose; decode them first
 * ({@code ffmpeg -i song.mp3 -ar 44100 -ac 2 song.wav}).
 */
public final class WavFile {

    private WavFile() {
    }

    public static AudioBuffer read(Path path) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN);
        if (buffer.remaining() < 12 || readTag(buffer) != 0x46464952 /* RIFF */) {
            throw new IOException(path + " is not a RIFF file");
        }
        buffer.getInt();                                       // RIFF size, unused
        if (readTag(buffer) != 0x45564157 /* WAVE */) {
            throw new IOException(path + " is not a WAVE file");
        }

        int formatTag = 0;
        int channels = 0;
        int bits = 0;
        int sampleRate = 0;
        byte[] data = null;

        while (buffer.remaining() >= 8) {
            int id = readTag(buffer);
            int size = buffer.getInt();
            if (size < 0 || size > buffer.remaining()) {
                size = buffer.remaining();
            }

            if (id == 0x20746d66 /* "fmt " */) {
                int start = buffer.position();
                if (size < 16) {
                    throw new IOException("truncated fmt chunk in " + path);
                }
                formatTag = buffer.getShort() & 0xFFFF;
                channels = buffer.getShort() & 0xFFFF;
                sampleRate = buffer.getInt();
                buffer.getInt();                               // byte rate
                buffer.getShort();                             // block align
                bits = buffer.getShort() & 0xFFFF;
                if (formatTag == 0xFFFE && size >= 26) {       // WAVE_FORMAT_EXTENSIBLE
                    formatTag = buffer.getShort(start + 24) & 0xFFFF;
                }
                buffer.position(start + size + (size & 1));
            } else if (id == 0x61746164 /* "data" */) {
                data = new byte[size];
                buffer.get(data);
                break;
            } else {
                buffer.position(buffer.position() + size + (size & 1));
            }
        }

        if (data == null || channels < 1 || bits == 0) {
            throw new IOException("no usable audio data in " + path);
        }

        int width = bits / 8;
        int frames = data.length / (width * channels);
        AudioBuffer audio = new AudioBuffer(channels, frames, sampleRate);
        ByteBuffer samples = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

        for (int frame = 0; frame < frames; frame++) {
            for (int c = 0; c < channels; c++) {
                int at = (frame * channels + c) * width;
                audio.samples()[c * frames + frame] = decode(samples, at, bits, formatTag);
            }
        }
        return audio;
    }

    private static float decode(ByteBuffer data, int at, int bits, int formatTag)
            throws IOException {
        if (formatTag == 3) {
            return switch (bits) {
                case 32 -> data.getFloat(at);
                case 64 -> (float) data.getDouble(at);
                default -> throw new IOException("unsupported float width " + bits);
            };
        }
        return switch (bits) {
            case 16 -> data.getShort(at) / 32768.0f;
            case 24 -> {
                int value = (data.get(at) & 0xFF) << 8
                        | (data.get(at + 1) & 0xFF) << 16
                        | data.get(at + 2) << 24;
                yield (value >> 8) / 8388608.0f;
            }
            case 32 -> data.getInt(at) / 2147483648.0f;
            case 8 -> ((data.get(at) & 0xFF) - 128) / 128.0f;  // 8-bit WAV is unsigned
            default -> throw new IOException("unsupported bit depth " + bits);
        };
    }

    private static int readTag(ByteBuffer buffer) {
        return buffer.getInt();
    }
}
