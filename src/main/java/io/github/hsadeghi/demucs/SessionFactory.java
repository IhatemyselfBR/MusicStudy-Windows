package io.github.hsadeghi.demucs;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.nio.file.Path;
import java.util.List;

public final class SessionFactory {

    public enum Device {
        AUTO, CPU, CUDA, TENSORRT, DIRECTML, COREML, ROCM;

        public static Device parse(String text) {
            return Device.valueOf(text.toUpperCase());
        }

        @Override
        public String toString() {
            return name().toLowerCase();
        }
    }

    public static final class Config {
        public Device device = Device.CPU;
        public int deviceId = 0;
        public int threads = 0;
        public boolean fp16 = false;
        public String cacheDirectory = null;
        public boolean verbose = false;
    }

    public record Opened(OrtSession session, Device device) {}

    private SessionFactory() {}

    public static List<Device> availableDevices() {
        return List.of(Device.CPU);
    }

    public static Opened create(Path modelPath, Config config)
            throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();

        try (OrtSession.SessionOptions options =
                     new OrtSession.SessionOptions()) {
            options.setOptimizationLevel(
                OrtSession.SessionOptions.OptLevel.ALL_OPT
            );
            options.setExecutionMode(
                OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL
            );
            options.setInterOpNumThreads(1);

            if (config.threads > 0) {
                options.setIntraOpNumThreads(config.threads);
            }

            return new Opened(
                env.createSession(modelPath.toString(), options),
                Device.CPU
            );
        }
    }
}
