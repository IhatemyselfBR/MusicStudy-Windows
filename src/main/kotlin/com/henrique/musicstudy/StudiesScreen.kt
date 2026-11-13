package com.henrique.musicstudy

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import com.google.gson.Gson
import com.google.gson.JsonParser
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URL

private val gson = Gson()
private val categories = listOf("Campo Harmônico", "Escalas", "Acordes")

private data class StudyItem(
    val name: String = "",
    val description: String = "",
    val image: String? = null
)

private data class StudyNote(
    val title: String = "",
    val content: String = ""
)

private fun dataFolder(): File {
    val root = File(
        System.getenv("LOCALAPPDATA") ?: System.getProperty("user.home"),
        "MusicStudy"
    )
    root.mkdirs()
    return root
}

@Composable
fun StudiesScreen(onClose: () -> Unit) {
    var section by remember { mutableStateOf("Estudos") }
    var category by remember { mutableStateOf(categories[0]) }
    var search by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("Carregando biblioteca...") }

    val shared = remember { mutableStateMapOf<String, List<StudyItem>>() }
    val local = remember { mutableStateMapOf<String, MutableList<StudyItem>>() }
    val notes = remember { mutableStateListOf<StudyNote>() }

    var editorOpen by remember { mutableStateOf(false) }
    var editingIndex by remember { mutableIntStateOf(-1) }
    var title by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }

    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            val folder = dataFolder()

            runCatching {
                val file = File(folder, "anotacoes.json")
                if (file.exists()) {
                    val type = object : TypeToken<List<StudyNote>>() {}.type
                    val saved: List<StudyNote> = gson.fromJson(file.readText(), type) ?: emptyList()
                    notes.addAll(saved)
                }
            }

            categories.forEach { cat ->
                runCatching {
                    val file = File(folder, "estudos-${categories.indexOf(cat)}.json")
                    if (file.exists()) {
                        val type = object : TypeToken<List<StudyItem>>() {}.type
                        val saved: List<StudyItem> = gson.fromJson(file.readText(), type) ?: emptyList()
                        local[cat] = saved.toMutableList()
                    }
                }
            }

            runCatching {
                val url = URL("https://ihatemyselfbr.github.io/musicstudy-estudos/dados.json")
                val json = url.openConnection().apply {
                    connectTimeout = 10000
                    readTimeout = 15000
                }.getInputStream().bufferedReader().use { it.readText() }

                val root = JsonParser.parseString(json).asJsonObject

                categories.forEach { cat ->
                    val array = root.getAsJsonArray(cat)
                    shared[cat] = array?.mapNotNull { element ->
                        runCatching {
                            val obj = element.asJsonObject
                            StudyItem(
                                obj.get("name")?.asString ?: "",
                                obj.get("description")?.asString ?: "",
                                obj.get("image")?.asString
                            )
                        }.getOrNull()
                    } ?: emptyList()
                }
                status = "Biblioteca compartilhada atualizada."
            }.onFailure {
                status = "Offline: mostrando os conteúdos salvos."
            }
        }
    }

    fun saveLocal() {
        val index = categories.indexOf(category)
        File(dataFolder(), "estudos-$index.json")
            .writeText(gson.toJson(local[category] ?: emptyList<StudyItem>()))
    }

    fun saveNotes() {
        File(dataFolder(), "anotacoes.json").writeText(gson.toJson(notes.toList()))
    }

    Column(
        Modifier.fillMaxSize().padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text("MusicStudy — Biblioteca e anotações", style = MaterialTheme.typography.h5)
            TextButton(onClick = onClose) { Text("Fechar") }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { section = "Estudos" }) { Text("Estudos") }
            Button(onClick = { section = "Anotações" }) { Text("Anotações") }
        }

        Divider()
        Text(status, style = MaterialTheme.typography.caption)

        if (section == "Estudos") {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                categories.forEach { cat ->
                    Button(
                        onClick = {
                            category = cat
                            search = ""
                        }
                    ) { Text(cat) }
                }
            }

            OutlinedTextField(
                value = search,
                onValueChange = { search = it },
                label = { Text("Pesquisar conteúdo") },
                modifier = Modifier.fillMaxWidth()
            )

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(category, style = MaterialTheme.typography.h6)
                Button(onClick = {
                    title = ""
                    description = ""
                    editingIndex = -1
                    editorOpen = true
                }) { Text("+ Adicionar") }
            }

            val all = (shared[category] ?: emptyList()) +
                (local[category] ?: emptyList())

            val filtered = all.filter {
                it.name.contains(search, true) ||
                    it.description.contains(search, true)
            }

            LazyColumn(
                Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(filtered) { item ->
                    Card(Modifier.fillMaxWidth(), elevation = 2.dp) {
                        Column(Modifier.padding(14.dp)) {
                            Text(item.name, style = MaterialTheme.typography.subtitle1)
                            if (item.description.isNotBlank()) {
                                Spacer(Modifier.height(5.dp))
                                Text(item.description)
                            }
                            Text(
                                if (item in (local[category] ?: emptyList()))
                                    "Conteúdo local"
                                else "Biblioteca compartilhada",
                                style = MaterialTheme.typography.caption
                            )
                        }
                    }
                }
            }

        } else {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("Minhas anotações", style = MaterialTheme.typography.h6)
                Button(onClick = {
                    title = ""
                    description = ""
                    editingIndex = -1
                    editorOpen = true
                }) { Text("+ Nova anotação") }
            }

            LazyColumn(
                Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(notes.toList()) { note ->
                    Card(Modifier.fillMaxWidth(), elevation = 2.dp) {
                        Column(Modifier.padding(14.dp)) {
                            Text(note.title, style = MaterialTheme.typography.subtitle1)
                            Spacer(Modifier.height(5.dp))
                            Text(note.content)
                            Spacer(Modifier.height(8.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                TextButton(onClick = {
                                    editingIndex = notes.indexOf(note)
                                    title = note.title
                                    description = note.content
                                    editorOpen = true
                                }) { Text("Editar") }

                                TextButton(onClick = {
                                    notes.remove(note)
                                    saveNotes()
                                }) { Text("Excluir") }
                            }
                        }
                    }
                }
            }
        }
    }

    if (editorOpen) {
        Dialog(onDismissRequest = { editorOpen = false }) {
            Surface(
                Modifier.fillMaxWidth().padding(8.dp),
                shape = MaterialTheme.shapes.medium,
                elevation = 8.dp
            ) {
                Column(
                    Modifier.padding(20.dp).verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text(
                        if (section == "Anotações") "Anotação" else "Novo conteúdo",
                        style = MaterialTheme.typography.h6
                    )

                    OutlinedTextField(
                        value = title,
                        onValueChange = { title = it },
                        label = { Text("Título ou nome") },
                        modifier = Modifier.fillMaxWidth()
                    )

                    OutlinedTextField(
                        value = description,
                        onValueChange = { description = it },
                        label = { Text("Descrição ou conteúdo") },
                        modifier = Modifier.fillMaxWidth().heightIn(min = 120.dp)
                    )

                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        TextButton(onClick = { editorOpen = false }) {
                            Text("Cancelar")
                        }
                        Button(onClick = {
                            if (title.isNotBlank()) {
                                if (section == "Anotações") {
                                    val note = StudyNote(title.trim(), description.trim())
                                    if (editingIndex >= 0 && editingIndex < notes.size) {
                                        notes[editingIndex] = note
                                    } else {
                                        notes.add(note)
                                    }
                                    saveNotes()
                                } else {
                                    val list = local.getOrPut(category) { mutableListOf() }
                                    list.add(StudyItem(title.trim(), description.trim()))
                                    saveLocal()
                                }
                                editorOpen = false
                            }
                        }) { Text("Salvar") }
                    }
                }
            }
        }
    }
}
