package com.example.bowlsmeasuring

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.activity.compose.setContent
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import kotlinx.coroutines.launch
import kotlin.math.hypot
import org.opencv.android.OpenCVLoader
import java.io.File
import androidx.core.content.FileProvider

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!OpenCVLoader.initLocal()) {
            throw IllegalStateException("Unable to initialise OpenCV")
        }

        setContent {
            MaterialTheme {
                Surface(Modifier.fillMaxSize()) {
                    BowlsMeasuringScreen()
                }
            }
        }
    }
}

enum class MeasureMode { NONE, JACK, WOOD }

data class RadarPoint(
    val label: String,
    val rank: Int,
    val relativeDistance: Double,
    val angleRad: Double,
    val isJack: Boolean
)
@Composable
private fun BowlsMeasuringScreen() {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var imageBitmap by remember { mutableStateOf<ImageBitmap?>(null) }
    val measurements = remember { mutableStateListOf<ObjectMeasurement>() }
    var mode by remember { mutableStateOf(MeasureMode.NONE) }
    var pendingCentre by remember { mutableStateOf<Point2?>(null) }
    var status by remember { mutableStateOf("Take or choose a photograph to begin.") }
    var resultText by remember { mutableStateOf("") }
    var jackRatio by remember { mutableFloatStateOf(0.55f) }
    var focalLength by remember { mutableFloatStateOf(650f) }
    var automaticDetections by remember {
        mutableStateOf<List<WoodDetection>>(emptyList())
    }
    var woodRankings by remember { mutableStateOf<Map<String, Int>>(emptyMap()) }
    var radarPoints by remember { mutableStateOf<List<RadarPoint>>(emptyList()) }

    // State variable to store the temporary high-res picture link destination
    var highResPhotoUri by remember { mutableStateOf<Uri?>(null) }

    // Helper to generate a secure FileProvider destination path in app cache space
    fun createTempPictureUri(): Uri {
        val tempFile = File.createTempFile("bowl_capture_", ".jpg", context.cacheDir).apply {
            createNewFile()
            deleteOnExit()
        }
        return FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            tempFile
        )
    }

    fun getNextAvailableWoodLetter(currentMeasurements: List<ObjectMeasurement>): Char {
        val existingLetters = currentMeasurements
            .filter { it.name.startsWith("Wood ") }
            .mapNotNull { it.name.removePrefix("Wood ").trim().firstOrNull() }
            .toSet()
        var c = 'a'
        while (existingLetters.contains(c)) {
            c++
            if (c > 'z') break
        }
        return c
    }

    fun loadBitmap(b: Bitmap) {
        bitmap = b
        imageBitmap = b.asImageBitmap()
        measurements.clear()
        automaticDetections = emptyList()
        woodRankings = emptyMap()
        pendingCentre = null
        mode = MeasureMode.JACK
        resultText = ""
        status = "Photo loaded. Tap the Jack."
    }

    fun loadUri(uri: Uri) {
        val bm = context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it)
        }
            ?.let(::loadBitmap)
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) loadUri(uri)
    }

    // FIXED: Swapped to full-resolution TakePicture contract hook
    val camera = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            highResPhotoUri?.let { uri -> loadUri(uri) }
        } else {
            status = "Camera capture cancelled or failed."
        }
    }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text("Bowls Measuring", style = MaterialTheme.typography.headlineSmall)
        Text(
            "Prototype v0.3 — semi-auto selection",
            style = MaterialTheme.typography.bodySmall
        )
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                // Generate tracking handle and request full canvas context frame write
                val targetUri = createTempPictureUri()
                highResPhotoUri = targetUri
                camera.launch(targetUri)
            }) { Text("Take Photo") }

            OutlinedButton(onClick = { picker.launch("image/*") }) { Text("Choose Photo") }
            OutlinedButton(onClick = {
                measurements.clear(); pendingCentre = null; mode = MeasureMode.NONE
                woodRankings = emptyMap(); resultText = ""; status = "Measurements cleared."
            }) { Text("Clear") }
        }

        OutlinedButton(
            enabled = bitmap != null,
            onClick = {
                val b = bitmap ?: return@OutlinedButton
                status = "Detecting circular objects..."
                scope.launch {
                    val woodDetectorResult = WoodDetector.detect(b)
                    automaticDetections = woodDetectorResult.detections
                    bitmap = woodDetectorResult.processedBitmap
                    imageBitmap = woodDetectorResult.processedBitmap.asImageBitmap()
                    status = "Detected ${automaticDetections.size} circular candidates."
                }
            }
        ) {
            Text("Detect Woods")
        }

        Text(status, style = MaterialTheme.typography.bodySmall)
        imageBitmap?.let { image ->
            MeasurementCanvas(
                image = image,
                measurements = measurements,
                detections = automaticDetections,
                pendingCentre = pendingCentre,
                rankings = woodRankings
            ) { point ->
                if (mode == MeasureMode.NONE) return@MeasurementCanvas

                status = "Fitting ellipse..."
                scope.launch {
                    val b = bitmap ?: return@launch
                    val detection = WoodDetector.detectNear(b, point)

                    if (detection != null) {
                        val corrected = Point2(detection.centre.x.toFloat(), detection.centre.y.toFloat())

                        if (mode == MeasureMode.JACK) {
                            measurements.removeAll { it.name == "Jack" }
                            measurements.add(ObjectMeasurement("Jack", corrected, detection.radiusPx))
                            mode = MeasureMode.WOOD
                            val nextL = getNextAvailableWoodLetter(measurements)
                            status = "Jack set. Wood $nextL: tap its centre."
                        } else {
                            val overlapIndex = measurements.indexOfFirst {
                                it.name.startsWith("Wood") && hypot((it.centre.x - corrected.x).toDouble(), (it.centre.y - corrected.y).toDouble()) < detection.radiusPx * 1.1f
                            }

                            if (overlapIndex != -1) {
                                val originalName = measurements[overlapIndex].name
                                measurements[overlapIndex] = ObjectMeasurement(originalName, corrected, detection.radiusPx)
                                status = "$originalName updated."
                            } else {
                                val nextLetter = getNextAvailableWoodLetter(measurements)
                                measurements.add(ObjectMeasurement("Wood $nextLetter", corrected, detection.radiusPx))
                                val nextNext = getNextAvailableWoodLetter(measurements)
                                status = "Wood $nextLetter set. Wood $nextNext: tap its centre."
                            }
                        }
                    } else {
                        status = "Fit failed. Try tapping the clear edge of the bowl."
                    }
                    pendingCentre = null
                }
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                mode = MeasureMode.JACK; pendingCentre = null; status =
                "Jack: tap near its centre."
            }) { Text("Set Jack") }
            Button(
                enabled = measurements.any { it.name == "Jack" },
                onClick = {
                    mode = MeasureMode.WOOD; pendingCentre = null
                    val nextL = getNextAvailableWoodLetter(measurements)
                    status = "Wood $nextL: tap near its centre."
                }) { Text("Add Wood") }
            Button(enabled = measurements.any { it.name == "Jack" } && measurements.count {
                it.name.startsWith("Wood")
            } >= 2, onClick = {
                val jack = measurements.first { it.name == "Jack" }
                val woods = measurements.filter { it.name.startsWith("Wood") }
                val corrected = SizeDepthModel.calculate(
                    jack, jackRatio.toDouble(), focalLength.toDouble(),
                    (bitmap?.width ?: 0) / 2.0, (bitmap?.height ?: 0) / 2.0, woods
                )
                if (corrected == null) {
                    resultText = "Model needs at least two woods."
                    status = "Calculation failed."
                } else {
                    woodRankings = corrected.measurements.associate { it.name to it.rank }

                    // BUILD DIAGNOSTIC RADAR SYNC DATA
                    val jack = measurements.first { it.name == "Jack" }
                    val tempRadarList = mutableListOf<RadarPoint>()

                    // Insert the Jack right in the center (distance 0)
                    tempRadarList.add(RadarPoint("Jack", 0, 0.0, 0.0, true))

                    corrected.measurements.forEach { correctedWood ->
                        // Locate the matching raw input measurement to evaluate its direction angle from the Jack
                        val rawInputWood = measurements.firstOrNull { it.name == correctedWood.name }
                        val angle = if (rawInputWood != null) {
                            // Compute image plane angle relative to the Jack position
                            kotlin.math.atan2(
                                (jack.centre.y - rawInputWood.centre.y).toDouble(),
                                (rawInputWood.centre.x - jack.centre.x).toDouble()
                            )
                        } else {
                            0.0
                        }

                        tempRadarList.add(
                            RadarPoint(
                                label = correctedWood.name.removePrefix("Wood ").trim(),
                                rank = correctedWood.rank,
                                relativeDistance = correctedWood.relativeDistance,
                                angleRad = angle,
                                isJack = false
                            )
                        )
                    }
                    radarPoints = tempRadarList // Triggers UI re-render

                    resultText = buildString {
                        // ... preserves your existing text appending code ...
                    }
                    mode = MeasureMode.NONE
                    status = "Calculation complete."
                }
            }) { Text("Calculate") }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text("Perspective model", style = MaterialTheme.typography.titleMedium)
                Text("Jack / wood diameter ratio: ${"%.2f".format(jackRatio)}")
                Slider(jackRatio, { jackRatio = it }, valueRange = .30f..1.0f)
                Text("Focal length approximation: ${focalLength.toInt()} px")
                Slider(focalLength, { focalLength = it }, valueRange = 200f..2000f)
            }
        }
        if (measurements.isNotEmpty()) Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text("Measurements", style = MaterialTheme.typography.titleMedium)
                measurements.sortedBy { m ->
                    if (m.name == "Jack") " " else m.name
                }.forEach {
                    val labelText = if (it.name == "Jack") "Jack" else "Wood ${it.name.removePrefix("Wood ").trim()}"
                    Text("$labelText: (${it.centre.x.toInt()}, ${it.centre.y.toInt()}) radius ${"%.1f".format(it.apparentRadiusPx)} px")
                }
            }
        }
        if (resultText.isNotBlank()) Card(Modifier.fillMaxWidth()) {
            Text(resultText, Modifier.padding(12.dp))
        }
        if (radarPoints.isNotEmpty()) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp)) {
                    Text(
                        "Calculated Bird's-Eye Perspective Map",
                        style = MaterialTheme.typography.titleMedium
                    )
                    Spacer(Modifier.height(8.dp))
                    BirdsEyeRadar(points = radarPoints)
                }
            }
        }

        if (resultText.isNotBlank()) {
            Card(Modifier.fillMaxWidth()) {
                Text(resultText, Modifier.padding(12.dp))
            }
        }
        Spacer(Modifier.height(20.dp))
    }
}

@Composable
private fun MeasurementCanvas(
    image: ImageBitmap,
    measurements: List<ObjectMeasurement>,
    detections: List<WoodDetection>,
    pendingCentre: Point2?,
    rankings: Map<String, Int>,
    onTap: (Point2) -> Unit
) {
    Box(Modifier
        .fillMaxWidth()
        .background(Color.Black)) {
        Canvas(
            Modifier
                .fillMaxWidth()
                .aspectRatio(image.width.toFloat() / image.height.toFloat())
                .pointerInput(image) {
                    detectTapGestures { offset ->
                        val sx = image.width.toFloat() / size.width
                        val sy = image.height.toFloat() / size.height
                        onTap(Point2(offset.x * sx, offset.y * sy))
                    }
                }) {
            drawImage(image, dstSize = IntSize(size.width.toInt(), size.height.toInt()))
            val sx = size.width / image.width.toFloat()
            val sy = size.height / image.height.toFloat()

            measurements.forEach { m ->
                val c = Offset(m.centre.x * sx, m.centre.y * sy)
                val r = m.apparentRadiusPx * (sx + sy) / 2f
                val col = if (m.name == "Jack") Color.Yellow else Color.Green
                drawCircle(col, r, c, style = Stroke(3f))
                drawCircle(col, 5f, c)

                val baseLabel = if (m.name == "Jack") "J" else m.name.removePrefix("Wood ").trim()
                val rank = rankings[m.name]
                val label = if (rank != null) "$baseLabel ($rank)" else baseLabel

                drawContext.canvas.nativeCanvas.drawText(label, c.x, c.y + 14f,
                    android.graphics.Paint().apply {
                        color = if (m.name == "Jack") android.graphics.Color.YELLOW else android.graphics.Color.GREEN
                        textSize = 44f
                        isAntiAlias = true
                        textAlign = android.graphics.Paint.Align.CENTER
                        typeface = android.graphics.Typeface.DEFAULT_BOLD
                        setShadowLayer(10f, 0f, 0f, android.graphics.Color.BLACK)
                    }
                )
            }

            detections.forEachIndexed { index, detection ->
                val alreadyMeasured = measurements.any {
                    val dx = it.centre.x - detection.centre.x
                    val dy = it.centre.y - detection.centre.y
                    kotlin.math.sqrt(dx * dx + dy * dy) < detection.radiusPx * 0.75f
                }

                if (!alreadyMeasured) {
                    val centre = Offset((detection.centre.x * sx).toFloat(), (detection.centre.y * sy).toFloat())
                    val radius = (detection.radiusPx * (sx + sy) / 2f).toFloat()
                    drawCircle(Color.Cyan, radius, centre, style = Stroke(2f))

                    val letterLabel = ('a' + index).toString()
                    drawContext.canvas.nativeCanvas.drawText(letterLabel, centre.x, centre.y + 10f,
                        android.graphics.Paint().apply {
                            color = android.graphics.Color.CYAN; textSize = 34f; isAntiAlias = true; textAlign = android.graphics.Paint.Align.CENTER; setShadowLayer(8f, 0f, 0f, android.graphics.Color.BLACK)
                        }
                    )
                }
            }
            pendingCentre?.let { drawCircle(Color.Red, 7f, Offset(it.x * sx, it.y * sy)) }
        }
    }
}

@Composable
private fun BirdsEyeRadar(
    points: List<RadarPoint>,modifier: Modifier = Modifier
) {
    Box(
        modifier
            .fillMaxWidth()
            .height(280.dp)
            .background(Color(0xFF1E251F)) // Deep dark grass background
            .padding(16.dp)
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val center = Offset(size.width / 2f, size.height / 2f)
            val maxRadarRadius = minOf(size.width, size.height) / 2f * 0.85f

            // 1. Draw Concentric Proximity Guide Rings
            drawCircle(Color.Gray.copy(alpha = 0.2f), maxRadarRadius * 0.33f, center, style = Stroke(1f))
            drawCircle(Color.Gray.copy(alpha = 0.2f), maxRadarRadius * 0.66f, center, style = Stroke(1f))
            drawCircle(Color.Gray.copy(alpha = 0.3f), maxRadarRadius, center, style = Stroke(2f))

            // Find the maximum relative distance to scale everything safely inside the canvas limits
            val maxDist = points.maxOfOrNull { it.relativeDistance }?.takeIf { it > 0 } ?: 1.0
            val scaleFactor = maxRadarRadius / maxDist

            // 2. Plot the Points
            points.forEach { pt ->
                // Convert polar coordinates (distance + angle) into top-down XY canvas offsets
                // We flip the Y component so that objects at the top of the photo render at the top of the radar
                val dx = (pt.relativeDistance * scaleFactor * kotlin.math.cos(pt.angleRad)).toFloat()
                val dy = -(pt.relativeDistance * scaleFactor * kotlin.math.sin(pt.angleRad)).toFloat()
                val targetPoint = Offset(center.x + dx, center.y + dy)

                val color = if (pt.isJack) Color.Yellow else Color.Green

                // Draw the object mark
                drawCircle(color, if (pt.isJack) 8f else 12f, targetPoint)
                if (!pt.isJack) drawCircle(Color.Black, 12f, targetPoint, style = Stroke(2f))

                // Render Identity Letter & Rank Text badge
                drawIntoCanvas { canvas ->
                    val paint = android.graphics.Paint().apply {
                        this.color = if (pt.isJack) android.graphics.Color.YELLOW else android.graphics.Color.GREEN
                        textSize = 34f
                        isFakeBoldText = true
                        textAlign = android.graphics.Paint.Align.CENTER
                    }
                    val text = if (pt.isJack) "J" else "${pt.label} (${pt.rank})"
                    canvas.nativeCanvas.drawText(text, targetPoint.x, targetPoint.y - 18f, paint)
                }
            }
        }
    }
}