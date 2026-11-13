// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * What {@code -m} resolved to: one graph, or several to be blended.
 *
 * <p>{@code htdemucs_ft}, {@code mdx}, {@code mdx_extra} and the quantised pair are <em>bags</em>
 * -- four independently trained networks whose stems are averaged with per-stem weights. Upstream
 * runs each of them over the whole track and adds up the results; because every step after the
 * network (ISTFT, cross-fade, denormalisation) is linear, the same answer comes out of blending
 * the raw graph outputs chunk by chunk, which is what {@link Separator} does.
 */
public record ModelBundle(String name, List<Member> members) {

    /** One network of a bag, and how much of each stem it contributes. */
    public record Member(Path path, float[] weights) {

        /** {@code null} weights mean an equal share of every stem. */
        public Member {
            weights = weights == null ? null : weights.clone();
        }
    }

    public boolean isBag() {
        return members.size() > 1;
    }

    /**
     * Turns a user-supplied {@code -m} into the list of graphs to open. Accepts a graph
     * ({@code htdemucs.onnx}), a bag manifest ({@code htdemucs_ft.json}), or either one with the
     * extension left off.
     */
    public static ModelBundle resolve(Path path) {
        String name = path.getFileName().toString();
        if (name.endsWith(".onnx")) {
            return single(path);
        }
        if (name.endsWith(".json")) {
            return fromManifest(path);
        }

        Path graph = path.resolveSibling(name + ".onnx");
        if (Files.exists(graph)) {
            return single(graph);
        }
        Path manifest = path.resolveSibling(name + ".json");
        if (Files.exists(manifest)) {
            return fromManifest(manifest);
        }
        throw new IllegalArgumentException("neither " + graph + " nor " + manifest + " exists");
    }

    private static ModelBundle single(Path path) {
        return new ModelBundle(Sidecar.stem(path), List.of(new Member(path, null)));
    }

    private static ModelBundle fromManifest(Path path) {
        Sidecar manifest = Sidecar.read(path);
        List<Sidecar> entries = manifest.objects("models");
        if (entries.isEmpty()) {
            // A plain sidecar, not a manifest: the graph sits next to it.
            return single(path.resolveSibling(Sidecar.stem(path) + ".onnx"));
        }

        List<Member> members = new ArrayList<>(entries.size());
        for (Sidecar entry : entries) {
            String file = entry.string("model").orElseThrow(() -> new IllegalArgumentException(
                    path + ": a bag entry has no \"model\""));
            Path member = path.resolveSibling(file);
            if (!Files.exists(member)) {
                throw new IllegalArgumentException(
                        path + " lists " + file + ", which is not beside it");
            }
            members.add(new Member(member, entry.numbers("weights").orElse(null)));
        }
        return new ModelBundle(manifest.string("name").orElse(Sidecar.stem(path)),
                List.copyOf(members));
    }

    /**
     * Normalised {@code [model][source]} weights, or {@code null} for a single graph.
     *
     * <p>Upstream sums {@code weight * estimate} over the bag and divides each stem by the total
     * weight it received; folding that division in here makes the per-chunk blend a plain weighted
     * sum.
     */
    public float[][] mixingMatrix(int sourceCount) {
        if (!isBag()) {
            return null;
        }

        float[][] matrix = new float[members.size()][sourceCount];
        for (int m = 0; m < members.size(); m++) {
            float[] row = members.get(m).weights();
            if (row == null) {
                java.util.Arrays.fill(matrix[m], 1.0f);      // an equal share of every stem
                continue;
            }
            if (row.length != sourceCount) {
                throw new IllegalArgumentException(members.get(m).path() + " carries " + row.length
                        + " weights but the model has " + sourceCount + " stems");
            }
            System.arraycopy(row, 0, matrix[m], 0, sourceCount);
        }

        for (int s = 0; s < sourceCount; s++) {
            float total = 0.0f;
            for (float[] row : matrix) {
                total += row[s];
            }
            if (total <= 0.0f) {
                throw new IllegalArgumentException(
                        "no model in the bag contributes to stem " + s);
            }
            for (float[] row : matrix) {
                row[s] /= total;
            }
        }
        return matrix;
    }
}
