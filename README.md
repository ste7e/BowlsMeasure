# Bowls Measuring — Android / Kotlin / Jetpack Compose v0.1.2

Android rework of the original C#/Avalonia prototype. Native Kotlin + Jetpack Compose.

The prototype can take a photo, choose an existing photo, manually mark the jack and up to four woods, record apparent sizes, and compare raw image-space ordering with the experimental size/depth-corrected ordering.

The current camera path uses Android's `TakePicturePreview` contract for a simple prototype. A later version should use CameraX for full-resolution capture and a controlled preview. Android's current stable CameraX release is 1.6.2.

The geometry deliberately remains the same experimental hypothesis as v0.1: apparent wood size is used as a relative camera-depth cue. The next mathematical step is an explicit ground-plane/camera model.

Open the folder in Android Studio and run it on an Android device/emulator.


## Build toolchain

This version uses Android Gradle Plugin 9.1.1 and is intended to be built with Gradle 9.3.1 and JDK 17. It compiles against Android API 37 to match the current Compose dependency set.
