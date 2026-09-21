package com.example.bowlsmeasuring

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.launch
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
import androidx.compose.ui.graphics.nativeCanvas
import kotlinx.coroutines.launch
import kotlin.math.hypot
import org.opencv.android.OpenCVLoader

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
    // New state to hold computed positional rankings for the overlay canvas
    var woodRankings by remember { mutableStateOf<Map<String, Int>>(emptyMap()) }

    // Helper to find the lowest available wood character letter for gapless sequence
    fun getNextAvailableWoodLetter(currentMeasurements: List<ObjectMeasurement>): Char {
        val existingLetters = currentMeasurements
            .filter { it.name.startsWith("Wood ") }
            .mapNotNull { it.name.removePrefix("Wood ").trim().firstOrNull() }
            .toSet()
        var c = 'a'
        while (existingLetters.contains(c)) {
            c++
            if (c > 'z') break // Safety anchor break
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
        // UX: Auto-start in Jack mode
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
    val camera =
        rememberLauncherForActivityResult(ActivityResultContracts.TakePicturePreview()) { b ->
            if (b != null) loadBitmap(b)
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
            Button(onClick = { camera.launch() }) { Text("Take Photo") }
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
                            mode = MeasureMode.WOOD // Auto-switch to continuous wood mode
                            val nextL = getNextAvailableWoodLetter(measurements)
                            status = "Jack set. Wood $nextL: tap its centre."
                        } else {
                            // Smart Overwrite: Check if we clicked an existing wood to refine it
                            val overlapIndex = measurements.indexOfFirst {
                                it.name.startsWith("Wood") && hypot((it.centre.x - corrected.x).toDouble(), (it.centre.y - corrected.y).toDouble()) < detection.radiusPx * 1.1f
                            }

                            if (overlapIndex != -1) {
                                // Update existing entry, keep original name/letter
                                val originalName = measurements[overlapIndex].name
                                measurements[overlapIndex] = ObjectMeasurement(originalName, corrected, detection.radiusPx)
                                status = "$originalName updated."
                            } else {
                                // Add new wood using gapless letter sequence
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
                    // Populate tracking state map to trigger overlay canvas position renders
                    woodRankings = corrected.measurements.associate { it.name to it.rank }
                    resultText = buildString {
                        appendLine("GROUND-PLANE CORRECTED ORDER")
                        appendLine(corrected.measurements.joinToString(" → ") { it.name.removePrefix("Wood ").trim() })
                        appendLine()
                        appendLine("RELATIVE DISTANCES")
                        corrected.measurements.forEach {
                            val letter = it.name.removePrefix("Wood ").trim()
                            appendLine("$letter (${it.rank})   ${"%.3f".format(it.relativeDistance)} × nearest")
                        }
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
                Slider(jackRatio, { jackRatio = it }, valueRange = .30f..0.80f)
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
    Box(Modifier.fillMaxWidth().background(Color.Black)) {
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

                // Clean alphabetical mapping: extracts 'a', 'b', 'c' instead of index values
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

                    // Unified pipeline alignment: sets automatic candidates as lowercase letters
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