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
import androidx.activity.compose.setContent
import androidx.compose.runtime.getValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import kotlinx.coroutines.launch
import kotlin.math.hypot
import org.opencv.android.OpenCVLoader
import java.io.File
import androidx.lifecycle.ViewModel
import android.content.Context
import android.graphics.BitmapFactory
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.runtime.mutableFloatStateOf

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // ... inside your onCreate function ...
        if (!OpenCVLoader.initLocal()) {
            throw IllegalStateException("Unable to initialise OpenCV")
        }

        // Native Activity-scoped provider (no extra libraries required)
        val viewModel = androidx.lifecycle.ViewModelProvider(this)[BowlsViewModel::class.java]

        setContent {
            MaterialTheme {
                Surface(Modifier.fillMaxSize()) {
                    // Pass the instantiated viewmodel here
                    BowlsMeasuringScreen(viewModel = viewModel)
                }
            }
        }
    }
}
class BowlsViewModel : ViewModel() {
    var bitmap by mutableStateOf<Bitmap?>(null)
    var imageBitmap by mutableStateOf<ImageBitmap?>(null)
    val measurements = mutableStateListOf<ObjectMeasurement>()
    var mode by mutableStateOf(MeasureMode.NONE)
    var pendingCentre by mutableStateOf<Point2?>(null)
    var status by mutableStateOf("Take or choose a photograph to begin.")
    var resultText by mutableStateOf("")
    var jackRatio by mutableFloatStateOf(0.55f)
    var focalLength by mutableFloatStateOf(650f)
    var automaticDetections by mutableStateOf<List<WoodDetection>>(emptyList())
    var woodRankings by mutableStateOf<Map<String, Int>>(emptyMap())
    var showPerspectiveGrid by mutableStateOf(false)
    var radarPoints by mutableStateOf<List<RadarPoint>>(emptyList())
    var highResPhotoUriString by mutableStateOf<String?>(null)
    var cameraTilt by mutableFloatStateOf(23f)// Initial camera tilt pitch in degrees
    var gridScale by mutableFloatStateOf(1.0f)   // Allows scaling the width of the red lines to match planks


    fun createTempPictureUri(context: Context): Uri {
        val tempFile = File.createTempFile("bowl_capture_", ".jpg", context.cacheDir).apply {
            createNewFile()
            deleteOnExit()
        }
        return androidx.core.content.FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            tempFile
        )
    }

    fun getNextAvailableWoodLetter(): Char {
        val existingLetters = measurements
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

    fun updateFocalLengthFromExif(context: Context, uri: Uri) {
        try {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                val exif = android.media.ExifInterface(stream)
                val focal35mm = exif.getAttributeInt(android.media.ExifInterface.TAG_FOCAL_LENGTH_IN_35MM_FILM, 0)

                if (focal35mm > 0) {
                    focalLength = (focal35mm.toFloat() * 1024f) / 36f
                    status = "Photo loaded. Auto-calibrated focal plane via EXIF (${focal35mm}mm equiv)."
                } else {
                    val focalPhysical = exif.getAttributeDouble(android.media.ExifInterface.TAG_FOCAL_LENGTH, 0.0)
                    if (focalPhysical > 0.0) {
                        val estimated35mm = focalPhysical * 6.0
                        focalLength = ((estimated35mm * 1024.0) / 36.0).toFloat()
                        status = "Photo loaded. Estimated focal plane via physical EXIF metrics."
                    } else {
                        focalLength = 650f
                        status = "Photo loaded. No EXIF metrics discovered, using baseline fallback."
                    }
                }
            }
        } catch (e: Exception) {
            focalLength = 650f
            status = "Photo loaded. Error accessing metadata headers, using baseline fallback."
        }
    }

    fun rotateBitmapIfRequired(context: Context, img: Bitmap, uri: Uri): Bitmap {
        return try {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                val exif = android.media.ExifInterface(stream)
                val orientation = exif.getAttributeInt(
                    android.media.ExifInterface.TAG_ORIENTATION,
                    android.media.ExifInterface.ORIENTATION_NORMAL
                )
                val matrix = android.graphics.Matrix()
                when (orientation) {
                    android.media.ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                    android.media.ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                    android.media.ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
                    else -> return img
                }
                Bitmap.createBitmap(img, 0, 0, img.width, img.height, matrix, true)
            } ?: img
        } catch (e: Exception) {
            img
        }
    }

    fun loadUri(context: Context, uri: Uri) {
        try {
            val inputStream = context.contentResolver.openInputStream(uri)
            if (inputStream == null) {
                status = "Error: System returned an empty input stream for the file."
                return
            }
            val bm = inputStream.use { BitmapFactory.decodeStream(it) }
            if (bm != null) {
                val orientedBitmap = rotateBitmapIfRequired(context, bm, uri)
                loadBitmap(orientedBitmap)
                updateFocalLengthFromExif(context, uri)
            } else {
                status = "Error: Failed to decode file data into a valid photo image."
            }
        } catch (e: Exception) {
            status = "Image Load Error: ${e.localizedMessage ?: "Unknown parsing failure."}"
            e.printStackTrace()
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
private fun BowlsMeasuringScreen(viewModel: BowlsViewModel) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) viewModel.loadUri(context, uri)
    }

    val camera = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            val savedUriStr = viewModel.highResPhotoUriString
            if (savedUriStr != null) {
                viewModel.loadUri(context, Uri.parse(savedUriStr))
            } else {
                viewModel.status = "Error: Photo taken successfully, but the file path anchor was lost."
            }
        } else {
            viewModel.status = "Camera Error: Capture was cancelled by user, or storage access was denied."
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
                try {
                    val targetUri = viewModel.createTempPictureUri(context)
                    viewModel.highResPhotoUriString = targetUri.toString()
                    camera.launch(targetUri)
                } catch (e: Exception) {
                    viewModel.status = "Failed to launch device camera: ${e.localizedMessage}"
                }
            }) { Text("Take Photo") }

            OutlinedButton(onClick = { picker.launch("image/*") }) { Text("Choose Photo") }
            OutlinedButton(onClick = {
                viewModel.measurements.clear()
                viewModel.pendingCentre = null
                viewModel.mode = MeasureMode.NONE
                viewModel.woodRankings = emptyMap()
                viewModel.resultText = ""
                viewModel.status = "Measurements cleared."
                viewModel.radarPoints = emptyList()
            }) { Text("Clear") }
        }

// Inside MainActivity.kt -> Look for the "Detect Woods" Button click handler:
        OutlinedButton(
            enabled = viewModel.bitmap != null,
            onClick = {
                val b = viewModel.bitmap ?: return@OutlinedButton
                viewModel.status = "Detecting circular objects via Peak Distance Transform..."
                scope.launch {
                    // 1. Execute our new peak blob detection pass
                    val blobDetections =
                        //SimpleBlobDetector.detect(b)
                        //RadialGradientAlignmentDetector.detect(b)
                        //ContrastWoodDetector.detect(b)
                        WoodDetector.detect(b)

                    // 2. Map the results cleanly over into your existing UI-compatible WoodDetection format
                    viewModel.automaticDetections = blobDetections.map { blob ->
                        // Map Point2 format directly across boundaries
                        val cvPoint = org.opencv.core.Point(blob.centre.x.toDouble(), blob.centre.y.toDouble())
                        WoodDetection(
                            centre = cvPoint,
                            radiusPx = blob.radiusPx,
                            circularity = blob.solidity,
                            confidence = blob.solidity
                        )
                    }

                    viewModel.status = "Detected ${viewModel.automaticDetections.size} stable circular objects."
                }
            }
        ) {
            Text("Detect Woods")
        }

        Text(viewModel.status, style = MaterialTheme.typography.bodySmall)

        viewModel.imageBitmap?.let { image ->
            MeasurementCanvas(
                image = image,
                measurements = viewModel.measurements,
                detections = viewModel.automaticDetections,
                pendingCentre = viewModel.pendingCentre,
                rankings = viewModel.woodRankings,
                showGrid = viewModel.showPerspectiveGrid,
                focalLength = viewModel.focalLength,
                cameraTilt = viewModel.cameraTilt,
                gridScale = viewModel.gridScale,
            ) { point ->
                if (viewModel.mode == MeasureMode.NONE) return@MeasurementCanvas

// Inside MainActivity.kt -> Locate the tap listener processing block inside MeasurementCanvas:
                viewModel.status = "Fitting ellipse using SBD Peak Fields..."
                scope.launch {
                    val b = viewModel.bitmap ?: return@launch

                    // 1. Invoke the new shadow-immune localized peak detector
                    val detection =
                        //SimpleBlobDetector.detectNear(b, point)
                        //RadialGradientAlignmentDetector.detectNear(b, point)
                        //ContrastWoodDetector.detectNear(b, point)
                        WoodDetector.detectNear(b, point)

                    if (detection != null) {
                        val corrected = Point2(detection.centre.x, detection.centre.y)

                        if (viewModel.mode == MeasureMode.JACK) {
                            viewModel.measurements.removeAll { it.name == "Jack" }
                            viewModel.measurements.add(ObjectMeasurement("Jack", corrected, detection.radiusPx))
                            viewModel.mode = MeasureMode.WOOD
                            val nextL = viewModel.getNextAvailableWoodLetter()
                            viewModel.status = "Jack set. Wood $nextL: tap its centre."
                        } else {
                            val overlapIndex = viewModel.measurements.indexOfFirst {
                                it.name.startsWith("Wood") && hypot((it.centre.x - corrected.x).toDouble(), (it.centre.y - corrected.y).toDouble()) < detection.radiusPx * 1.1f
                            }

                            if (overlapIndex != -1) {
                                val originalName = viewModel.measurements[overlapIndex].name
                                viewModel.measurements[overlapIndex] = ObjectMeasurement(originalName, corrected, detection.radiusPx)
                                viewModel.status = "$originalName updated."
                            } else {
                                val nextLetter = viewModel.getNextAvailableWoodLetter()
                                viewModel.measurements.add(ObjectMeasurement("Wood $nextLetter", corrected, detection.radiusPx))
                                val nextNext = viewModel.getNextAvailableWoodLetter()
                                viewModel.status = "Wood $nextLetter set. Wood $nextNext: tap its centre."
                            }
                        }
                    } else {
                        viewModel.status = "Fit failed. Make sure to tap directly inside the body of the bowl."
                    }
                    viewModel.pendingCentre = null
                }            }
        }

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                viewModel.mode = MeasureMode.JACK
                viewModel.pendingCentre = null
                viewModel.status = "Jack: tap near its centre."
            }) { Text("Set Jack") }

            Button(
                enabled = viewModel.measurements.any { it.name == "Jack" },
                onClick = {
                    viewModel.mode = MeasureMode.WOOD
                    viewModel.pendingCentre = null
                    val nextL = viewModel.getNextAvailableWoodLetter()
                    viewModel.status = "Wood $nextL: tap near its centre."
                }) { Text("Add Wood") }

            Button(
                enabled = viewModel.measurements.any { it.name == "Jack" } && viewModel.measurements.count { it.name.startsWith("Wood") } >= 2,
                onClick = {
                    val jack = viewModel.measurements.first { it.name == "Jack" }
                    val woods = viewModel.measurements.filter { it.name.startsWith("Wood") }
                    val corrected = SizeDepthModel.calculate(
                        jack, viewModel.jackRatio.toDouble(), viewModel.focalLength.toDouble(),
                        (viewModel.bitmap?.width ?: 0) / 2.0, (viewModel.bitmap?.height ?: 0) / 2.0, woods
                    )
                    if (corrected == null) {
                        viewModel.resultText = "Model needs at least two woods."
                        viewModel.status = "Calculation failed."
                    } else {
                        viewModel.woodRankings = corrected.measurements.associate { it.name to it.rank }

                        val jackObj = viewModel.measurements.first { it.name == "Jack" }
                        val tempRadarList = mutableListOf<RadarPoint>()
                        tempRadarList.add(RadarPoint("Jack", 0, 0.0, 0.0, true))

                        corrected.measurements.forEach { correctedWood ->
                            val rawInputWood = viewModel.measurements.firstOrNull { it.name == correctedWood.name }
                            val angle = if (rawInputWood != null) {
                                kotlin.math.atan2(
                                    (jackObj.centre.y - rawInputWood.centre.y).toDouble(),
                                    (rawInputWood.centre.x - jackObj.centre.x).toDouble()
                                )
                            } else 0.0

                            tempRadarList.add(RadarPoint(
                                label = correctedWood.name.removePrefix("Wood ").trim(),
                                rank = correctedWood.rank,
                                relativeDistance = correctedWood.relativeDistance,
                                angleRad = angle,
                                isJack = false
                            ))
                        }
                        viewModel.radarPoints = tempRadarList

                        viewModel.resultText = buildString {
                            appendLine("GROUND-PLANE CORRECTED ORDER")
                            appendLine(corrected.measurements.joinToString(" → ") { it.name.removePrefix("Wood ").trim() })
                            appendLine()
                            appendLine("RELATIVE DISTANCES")
                            corrected.measurements.forEach {
                                val letter = it.name.removePrefix("Wood ").trim()
                                appendLine("$letter (${it.rank})   ${"%.3f".format(it.relativeDistance)} × nearest")
                            }
                        }
                        viewModel.mode = MeasureMode.NONE
                        viewModel.status = "Calculation complete."
                    }
                }
            ) { Text("Calculate") }
        }

        Card(Modifier.fillMaxWidth()) {
            Row(
                Modifier.fillMaxWidth().padding(12.dp),
                verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("Show Perspective Grid Lines", style = MaterialTheme.typography.bodyMedium)
                Switch(
                    checked = viewModel.showPerspectiveGrid,
                    onCheckedChange = { viewModel.showPerspectiveGrid = it }
                )
            }
        }

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text("Perspective Model Calibration", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(4.dp))
                Text("Jack / wood diameter ratio: ${"%.2f".format(viewModel.jackRatio)}")
                Slider(viewModel.jackRatio, { viewModel.jackRatio = it }, valueRange = .30f..1.0f)

                Text(
                    text = "Focal Length: ${viewModel.focalLength.toInt()} px (Field of View Width)",
                    style = MaterialTheme.typography.bodyMedium
                )
                Slider(
                    value = viewModel.focalLength,
                    onValueChange = { viewModel.focalLength = it },
                    valueRange = 200f..2000f  // Standard smartphone lens threshold bounds
                )

                // NEW: Camera Tilt Control handles horizon height position independently
                Text(
                    text = "Camera Tilt: ${"%.1f".format(viewModel.cameraTilt)}° (Horizon Position)",
                    style = MaterialTheme.typography.bodyMedium
                )
                Slider(
                    value = viewModel.cameraTilt,
                    onValueChange = { viewModel.cameraTilt = it },
                    valueRange = 5f..60f
                )

                // NEW: Alignment scale handles spacing matches over ground patterns
                Text(
                    text = "Grid Plank Spacing: ${"%.2f".format(viewModel.gridScale)}",
                    style = MaterialTheme.typography.bodyMedium
                )
                Slider(
                    value = viewModel.gridScale,
                    onValueChange = { viewModel.gridScale = it },
                    valueRange = 0.2f..3.0f
                )

                Text(
                    text = "Tip: Lock Focal Length near the auto-extracted EXIF value (~654px for 23mm lenses). Adjust Camera Tilt until the red lines run parallel inside your floorboards.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                )
            }
        }

        if (viewModel.measurements.isNotEmpty()) Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text("Measurements", style = MaterialTheme.typography.titleMedium)
                viewModel.measurements.sortedBy { m ->
                    if (m.name == "Jack") " " else m.name
                }.forEach {
                    val labelText = if (it.name == "Jack") "Jack" else "Wood ${it.name.removePrefix("Wood ").trim()}"
                    Text("$labelText: (${it.centre.x.toInt()}, ${it.centre.y.toInt()}) radius ${"%.1f".format(it.apparentRadiusPx)} px")
                }
            }
        }

        if (viewModel.radarPoints.isNotEmpty()) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp)) {
                    Text("Calculated Bird's-Eye Perspective Map", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(8.dp))
                    BirdsEyeRadar(points = viewModel.radarPoints)
                }
            }
        }

        if (viewModel.resultText.isNotBlank()) Card(Modifier.fillMaxWidth()) {
            Text(viewModel.resultText, Modifier.padding(12.dp))
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
    showGrid: Boolean,
    focalLength: Float,
    cameraTilt: Float,
    gridScale: Float,
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

            // For mapping detected coordinates (which are always bound to the active image bitmap coordinates)
            val sx = size.width / image.width.toFloat()
            val sy = size.height / image.height.toFloat()

            // 1. TRUE RESOLUTION-INDEPENDENT 3D PERSPECTIVE GROUND-PLANE PROJECTION GRID
            if (showGrid) {
                val centerW = size.width / 2f
                val centerH = size.height / 2f

                // CRITICAL FIX: Scale the focal length strictly based on the UI Canvas screen dimensions
                // normalized against our 1024px reference frame. This keeps the grid perfectly static
                // regardless of whether the high-res original or the 1024px blurred matrix is displayed.
                val canvasMaxDim = maxOf(size.width, size.height)
                val f = focalLength * (canvasMaxDim / 1024f)

                // Physical Trigonometry Matrix
                val pitchRad = Math.toRadians(cameraTilt.toDouble()).toFloat()
                val cosP = kotlin.math.cos(pitchRad)
                val sinP = kotlin.math.sin(pitchRad)
                val H = 1.4f // Baseline standing camera height above the ground plane in meters

                // Absolute vanishing point height determined strictly by physics
                val vanishingPointY = centerH - f * (sinP / cosP)

                // 3D Matrix ground projection transformer mapping world (X, Z) to Screen coordinates
                val projectGroundPoint = { x: Float, z: Float ->
                    val camY = -H * cosP + z * sinP
                    val camZ = H * sinP + z * cosP
                    val screenX = centerW + (f * x / camZ)
                    val screenY = centerH - (f * camY / camZ)
                    Offset(screenX, screenY)
                }

                // Draw Longitudinal lines (Parallel floorboard boundaries extending to horizon)
                val lineCount = 6
                for (i in -lineCount..lineCount) {
                    val worldX = i * 0.25f * gridScale
                    // Project ray originating near the camera foot up into spatial infinity
                    val startPoint = projectGroundPoint(worldX, 0.8f)
                    val endPoint = Offset(centerW, vanishingPointY)

                    drawLine(
                        color = Color.Red.copy(alpha = 0.5f),
                        start = startPoint,
                        end = endPoint,
                        strokeWidth = 3f
                    )
                }

                // Draw Transverse lines (Horizontal depth indicators displaying compression scaling)
                val transverseSteps = 10
                for (i in 1..transverseSteps) {
                    val worldZ = 0.8f + (i * i * 0.25f * gridScale)
                    val gridY = projectGroundPoint(0f, worldZ).y

                    if (gridY in 0f..size.height) {
                        drawLine(
                            color = Color.Red.copy(alpha = 0.35f),
                            start = Offset(0f, gridY),
                            end = Offset(size.width, gridY),
                            strokeWidth = 2f
                        )
                    }
                }
            }

            // 2. EXISTING OBJECT BOUNDS RENDER PASSES
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