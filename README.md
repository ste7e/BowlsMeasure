# Bowls Measuring — Android / Kotlin / Jetpack Compose v0.2

Android prototype for measuring the relative distances of bowls from the jack in an oblique photograph.

## v0.2 geometry experiment

The manual marking workflow is retained: mark the jack centre/edge, then the centre/edge of up to four woods. The geometry now converts each apparent bowl radius into a relative camera-depth cue and fits all measured object centres to a common ground plane. Distances are measured after projecting the points onto that plane.

This is deliberately still an experimental model rather than calibrated photogrammetry. The focal length and jack/wood diameter ratio remain adjustable parameters. The UI reports the RMS distance of the measured 3-D points from the fitted plane so we can see how well the measurements support the model.

The camera path still uses Android's `TakePicturePreview` for the prototype. A later step should move to CameraX for full-resolution capture.

## Build toolchain

- Android Gradle Plugin 9.1.1
- Gradle 9.3.1
- JDK 17
- compileSdk / targetSdk 37
- Kotlin 2.2.20 + Compose compiler plugin
- Jetpack Compose BOM 2026.09.00

AGP 9 no longer requires the `org.jetbrains.kotlin.android` plugin, so it is intentionally absent.
