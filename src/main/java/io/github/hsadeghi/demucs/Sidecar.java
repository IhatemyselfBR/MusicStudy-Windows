// Copyright (c) 2026 Hamidreza Sadeghi
// SPDX-License-Identifier: MIT

package io.github.hsadeghi.demucs;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A deliberately small reader for the JSON written by {@code tools/export_onnx.py}.
 *
 * <p>Those files are machine-generated and nearly flat, so finding a top-level key and reading a
 * string, a number, an array of either, or -- for a bag manifest -- an array of objects is enough.
 * Anything it does not understand is ignored and the caller keeps its defaults.
 */
record Sidecar(String text) {

    /** The sidecar next to a graph, or an empty one if there is none. */
    static Sidecar beside(Path modelPath) {
        return read(modelPath.resolveSibling(stem(modelPath) + ".json"));
    }

    /** The named file, or an empty sidecar if it cannot be read. */
    static Sidecar read(Path path) {
        try {
            return new Sidecar(new String(Files.readAllBytes(path), StandardCharsets.UTF_8));
        } catch (IOException error) {
            return new Sidecar("");
        }
    }

    /** A file name with its extension removed. */
    static String stem(Path path) {
        String name = path.getFileName().toString();
        int dot = name.lastIndexOf('.');
        return dot <= 0 ? name : name.substring(0, dot);
    }

    Optional<String> string(String key) {
        Matcher matcher = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"").matcher(text);
        return matcher.find() ? Optional.of(matcher.group(1)) : Optional.empty();
    }

    Optional<Integer> integer(String key) {
        Matcher matcher = Pattern.compile("\"" + key + "\"\\s*:\\s*(-?\\d+)").matcher(text);
        return matcher.find() ? Optional.of(Integer.parseInt(matcher.group(1))) : Optional.empty();
    }

    Optional<List<String>> strings(String key) {
        return array(key).map(body -> {
            List<String> values = new ArrayList<>();
            Matcher items = Pattern.compile("\"([^\"]*)\"").matcher(body);
            while (items.find()) {
                values.add(items.group(1));
            }
            return values;
        });
    }

    Optional<float[]> numbers(String key) {
        return array(key).map(body -> {
            List<Float> values = new ArrayList<>();
            Matcher items = Pattern.compile("-?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?").matcher(body);
            while (items.find()) {
                values.add(Float.parseFloat(items.group()));
            }
            float[] out = new float[values.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = values.get(i);
            }
            return out;
        });
    }

    /**
     * An array of objects, each handed back as a sidecar of its own. Used for the {@code models}
     * list of a bag manifest, whose entries carry a nested array, so this has to track brace depth
     * rather than just look for the next closing brace.
     */
    List<Sidecar> objects(String key) {
        int start = valuePosition(key);
        if (start < 0 || text.charAt(start) != '[') {
            return List.of();
        }

        List<Sidecar> items = new ArrayList<>();
        int depth = 0;
        int from = 0;
        boolean inString = false;
        for (int i = start + 1; i < text.length(); i++) {
            char c = text.charAt(i);
            if (inString) {
                if (c == '\\') {
                    i++;
                } else if (c == '"') {
                    inString = false;
                }
            } else if (c == '"') {
                inString = true;
            } else if (c == '{') {
                if (depth++ == 0) {
                    from = i;
                }
            } else if (c == '}') {
                if (--depth == 0) {
                    items.add(new Sidecar(text.substring(from, i + 1)));
                }
            } else if (c == ']' && depth == 0) {
                break;
            }
        }
        return items;
    }

    /** The contents of a flat array, without its brackets. */
    private Optional<String> array(String key) {
        Matcher matcher = Pattern.compile("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]").matcher(text);
        return matcher.find() ? Optional.of(matcher.group(1)) : Optional.empty();
    }

    /** Index of the first non-space character after {@code "key":}, or -1. */
    private int valuePosition(String key) {
        Matcher matcher = Pattern.compile("\"" + key + "\"\\s*:\\s*").matcher(text);
        return matcher.find() ? matcher.end() : -1;
    }
}
