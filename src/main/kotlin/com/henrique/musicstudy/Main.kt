package com.henrique.musicstudy

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import io.github.hsadeghi.demucs.Separator
import io.github.hsadeghi.demucs.SessionFactory
import io.github.hsadeghi.demucs.audio.AudioBuffer
import io.github.hsadeghi.demucs.audio.SincResampler
import io.github.hsadeghi.demucs.audio.WavFile
import io.github.hsadeghi.demucs.audio.WavWriter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.awt.FileDialog
import java.awt.Frame
import java.io.File
import java.net.URL
import java.nio.file.Files
import java.nio.file.StandardCopyOption

private const val MODEL_URL =
    "https://huggingface.co/HRSadeghi/demucs-onnx/resolve/main/htdemucs.onnx"

private const val CONFIG_URL =
    "https://huggingface.co/HRSadeghi/demucs-onnx/resolve/main/htdemucs.json"

private fun chooseWav(): File? {
    val dialog = FileDialog(null as Frame?, "Selecionar áudio WAV", FileDialog.LOAD)
    dialog.isVisible = true
    return if (dialog.file != null) File(dialog.directory, dialog.file) else null
}

private fun download(url: String, destination: File) {
    destination.parentFile?.mkdirs()
    val temp = File(destination.parentFile, destination.name + ".part")

    try {
        URL(url).openStream().use { input ->
            temp.outputStream().buffered().use { output ->
                input.copyTo(output, 1024 * 256)
            }
        }

        require(temp.length() > 10) { "Download incompleto." }

        Files.move(
            temp.toPath(),
            destination.toPath(),
            StandardCopyOption.REPLACE_EXISTING
        )
    } catch (e: Exception) {
        temp.delete()
        throw IllegalStateException("Falha no download: ${e.message}", e)
    }
}

private fun separate(
    input: File,
    status: (String) -> Unit,
    progress: (Int) -> Unit
): List<File> {
    val home = File(
        System.getenv("LOCALAPPDATA") ?: System.getProperty("user.home"),
        "MusicStudy"
    )

    val modelDir = File(home, "demucs")
    val model = File(modelDir, "htdemucs.onnx")
    val config = File(modelDir, "htdemucs.json")

    status("Verificando modelo...")
    if (!model.exists() || model.length() < 50_000_000L) {
        download(MODEL_URL, model)
    }
    if (!config.exists() || config.length() < 10) {
        download(CONFIG_URL, config)
    }

    status("Lendo áudio WAV...")
    var audio: AudioBuffer = WavFile.read(input.toPath())

    require(audio.frames() > 0) { "O áudio está vazio." }

    audio = audio.remix(2)
    if (audio.sampleRate() != 44100) {
        audio = SincResampler.resample(audio, 44100)
    }

    val outputRoot = File(home, "Separadas")
    outputRoot.mkdirs()

    val project = File(
        outputRoot,
        "${input.nameWithoutExtension}_${System.currentTimeMillis()}"
    )
    check(project.mkdirs()) { "Não foi possível criar a pasta de saída." }

    status("Carregando inteligência artificial...")

    val options = Separator.Options().apply {
        batch = 1
        pipeline = false
        session.device = SessionFactory.Device.CPU
        session.threads = Runtime.getRuntime()
            .availableProcessors().coerceIn(1, 8)
    }

    val results = mutableListOf<File>()

    Separator(model.toPath(), options).use { separator ->
        val channels = separator.config().channels()
        val sources = separator.config().sources()

        val writers = sources.map { source ->
            val file = File(
                project,
                source.replace(Regex("[^a-zA-Z0-9_-]"), "_") + ".wav"
            )
            results.add(file)

            WavWriter(
                file.toPath(),
                separator.config().sampleRate(),
                channels,
                WavWriter.Format.INT16
            )
        }

        try {
            status("Separando faixas com Demucs...")

            separator.separate(
                audio.samples(),
                audio.frames(),
                { _, count, block ->
                    for (s in writers.indices) {
                        writers[s].write(
                            block,
                            s * channels * count,
                            count,
                            count
                        )
                    }
                },
                { done, total ->
                    progress(
                        if (total > 0) done * 100 / total else 0
                    )
                }
            )
        } finally {
            writers.forEach { it.close() }
        }
    }

    progress(100)
    return results
}

fun main() = application {
    Window(
        onCloseRequest = ::exitApplication,
        title = "MusicStudy",
    ) {
        var selected by remember { mutableStateOf<File?>(null) }
        var status by remember { mutableStateOf("Pronto para começar.") }
        var progress by remember { mutableStateOf(0) }
        var running by remember { mutableStateOf(false) }
        var result by remember { mutableStateOf<List<File>>(emptyList()) }

        val scope = rememberCoroutineScope()

        MaterialTheme {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(28.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text("MusicStudy", fontSize = 32.sp)
                Text("Estudos e ferramentas musicais")

                Divider()

                Text("SEPARAÇÃO DE ÁUDIO", fontSize = 13.sp)

                Button(
                    enabled = !running,
                    onClick = { selected = chooseWav() }
                ) {
                    Text("Selecionar arquivo WAV")
                }

                Text(selected?.absolutePath ?: "Nenhum áudio selecionado.")

                Button(
                    enabled = selected != null && !running,
                    onClick = {
                        val file = selected ?: return@Button
                        running = true
                        progress = 0
                        result = emptyList()

                        scope.launch(Dispatchers.IO) {
                            try {
                                val files = separate(
                                    file,
                                    { status = it },
                                    { progress = it }
                                )
                                result = files
                                status = "Separação concluída!"
                            } catch (e: Exception) {
                                status = "Erro: ${e.message}"
                            } finally {
                                running = false
                            }
                        }
                    }
                ) {
                    Text(if (running) "Processando..." else "Separar com Demucs")
                }

                LinearProgressIndicator(
                    progress = progress / 100f,
                    modifier = Modifier.fillMaxWidth()
                )

                Text("$progress%")
                Text(status)

                if (result.isNotEmpty()) {
                    Divider()
                    Text("FAIXAS GERADAS", fontSize = 13.sp)
                    result.forEach { Text(it.absolutePath) }
                }

                Spacer(Modifier.height(20.dp))
                Text("MusicStudy Desktop 2.1.0")
                Text("Processamento local com ONNX Runtime.")
            }
        }
    }
}
