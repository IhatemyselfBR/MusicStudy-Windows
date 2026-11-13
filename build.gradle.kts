plugins {
    kotlin("jvm") version "2.2.21"
    id("org.jetbrains.compose") version "1.9.3"
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.21"
}

group = "com.henrique.musicstudy"
version = "2.1.0"

repositories {
    google()
    mavenCentral()
}

dependencies {
    implementation(compose.desktop.currentOs)
    implementation("com.microsoft.onnxruntime:onnxruntime:1.31.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
}

kotlin {
    jvmToolchain(21)
}

compose.desktop {
    application {
        mainClass = "com.henrique.musicstudy.MainKt"

        nativeDistributions {
            targetFormats(
                org.jetbrains.compose.desktop.application.dsl.TargetFormat.Exe,
                org.jetbrains.compose.desktop.application.dsl.TargetFormat.Msi
            )
            packageName = "MusicStudy"
            packageVersion = "2.1.0"
            description = "MusicStudy - Estudos e ferramentas musicais"
            vendor = "MusicStudy"
        }
    }
}
