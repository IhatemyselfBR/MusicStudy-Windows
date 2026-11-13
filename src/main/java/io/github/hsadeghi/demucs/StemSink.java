// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs;

import java.io.IOException;

/**
 * Receives finalised output. {@code block} is {@code [sources][channels][count]}, planar, starting
 * at absolute sample {@code offset}. Spans arrive in order, are contiguous and never overlap.
 */
@FunctionalInterface
public interface StemSink {
    void accept(long offset, int count, float[] block) throws IOException;
}
