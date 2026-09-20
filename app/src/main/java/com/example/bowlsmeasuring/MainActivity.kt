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
import kotlin.math.hypot

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState); setContent { MaterialTheme { Surface(Modifier.fillMaxSize()) { BowlsMeasuringScreen() } } }
    }
}

private enum class MeasureMode { NONE, JACK, WOOD }

@Composable
private fun BowlsMeasuringScreen() {
    val context = androidx.compose.ui.platform.LocalContext.current
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var imageBitmap by remember { mutableStateOf<ImageBitmap?>(null) }
    val measurements = remember { mutableStateListOf<ObjectMeasurement>() }
    var mode by remember { mutableStateOf(MeasureMode.NONE) }
    var pendingCentre by remember { mutableStateOf<Point2?>(null) }
    var woodNumber by remember { mutableIntStateOf(1) }
    var status by remember { mutableStateOf("Take or choose a photograph to begin.") }
    var resultText by remember { mutableStateOf("") }
    var jackRatio by remember { mutableFloatStateOf(0.55f) }
    var focalLength by remember { mutableFloatStateOf(650f) }

    fun loadBitmap(b: Bitmap) {
        bitmap = b; imageBitmap = b.asImageBitmap(); measurements.clear(); pendingCentre =
            null; woodNumber = 1; mode = MeasureMode.NONE; resultText = ""; status =
            "Photo loaded: ${b.width} × ${b.height}"
    }

    fun loadUri(uri: Uri) {
        context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
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
            "Prototype v0.2 — size/depth cue + fitted ground plane",
            style = MaterialTheme.typography.bodySmall
        )
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { camera.launch() }) { Text("Take Photo") }
            OutlinedButton(onClick = { picker.launch("image/*") }) { Text("Choose Photo") }
            OutlinedButton(onClick = {
                measurements.clear(); pendingCentre = null; woodNumber = 1; mode =
                MeasureMode.NONE; resultText = ""; status = "Measurements cleared."
            }) { Text("Clear") }
        }
        Text(status, style = MaterialTheme.typography.bodySmall)
        imageBitmap?.let { image ->
            MeasurementCanvas(image, measurements, pendingCentre) { point ->
                if (mode == MeasureMode.NONE) return@MeasurementCanvas
                if (pendingCentre == null) {
                    pendingCentre = point; status =
                        if (mode == MeasureMode.JACK) "Jack centre set — tap its edge." else "Wood $woodNumber centre set — tap its edge."
                } else {
                    val c = pendingCentre!!;
                    val r = hypot((point.x - c.x).toDouble(), (point.y - c.y).toDouble()).toFloat()
                    if (r < 5f) {
                        status =
                            "That radius is too small. Tap the object edge."; return@MeasurementCanvas
                    }
                    if (mode == MeasureMode.JACK) {
                        measurements.removeAll { it.name == "Jack" }; measurements.add(
                            ObjectMeasurement("Jack", c, r)
                        ); status = "Jack recorded. Add the woods."
                    } else {
                        measurements.add(
                            ObjectMeasurement(
                                "Wood $woodNumber",
                                c,
                                r
                            )
                        ); woodNumber++; status = "Wood recorded."
                    }
                    pendingCentre = null; mode = MeasureMode.NONE
                }
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                mode = MeasureMode.JACK; pendingCentre = null; status =
                "Jack: tap centre, then edge."
            }) { Text("Set Jack") }
            Button(
                enabled = measurements.count { it.name.startsWith("Wood") } < 4,
                onClick = {
                    mode = MeasureMode.WOOD; pendingCentre = null; status =
                    "Wood $woodNumber: tap centre, then edge."
                }) { Text("Add Wood") }
            Button(enabled = measurements.any { it.name == "Jack" } && measurements.count {
                it.name.startsWith(
                    "Wood"
                )
            } >= 2, onClick = {
                val jack = measurements.first { it.name == "Jack" };
                val woods = measurements.filter { it.name.startsWith("Wood") }
                val raw = woods.sortedBy {
                    hypot(
                        (it.centre.x - jack.centre.x).toDouble(),
                        (it.centre.y - jack.centre.y).toDouble()
                    )
                }
                val corrected = SizeDepthModel.calculate(
                    jack,
                    jackRatio.toDouble(),
                    focalLength.toDouble(),
                    (bitmap?.width ?: 0) / 2.0,
                    (bitmap?.height ?: 0) / 2.0,
                    woods
                )
                if (corrected == null) {
                    resultText =
                        "The ground-plane model needs at least two woods plus the jack. Add another wood and calculate again."
                    status = "Not enough points for a plane fit."
                } else {
                    resultText = buildString {
                        appendLine("RAW IMAGE ORDER")
                        appendLine(raw.joinToString(" → ") { it.name })
                        appendLine()
                        appendLine("GROUND-PLANE CORRECTED ORDER")
                        appendLine(corrected.measurements.joinToString(" → ") { it.name })
                        appendLine()
                        appendLine("RELATIVE CORRECTED DISTANCES")
                        corrected.measurements.forEach {
                            appendLine(
                                "${it.rank}. ${it.name}   ${
                                    "%.3f".format(
                                        it.relativeDistance
                                    )
                                } × nearest"
                            )
                        }
                        appendLine()
                        appendLine("PLANE FIT RMS: ${"%.4f".format(corrected.planeRms)}")
                        appendLine("(smaller means the measured centres/sizes lie more closely on one plane)")
                    }
                    status = "Ground-plane calculation complete."
                }
            }) { Text("Calculate") }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text(
                    "Perspective model",
                    style = MaterialTheme.typography.titleMedium
                ); Text("Jack / wood diameter ratio: ${"%.2f".format(jackRatio)}"); Slider(
                jackRatio,
                { jackRatio = it },
                valueRange = .30f..0.80f
            ); Text("Focal length approximation: ${focalLength.toInt()} px"); Slider(
                focalLength,
                { focalLength = it },
                valueRange = 200f..2000f
            ); Text("Experimental parameters. The next version should use an explicit ground-plane model.")
            }
        }
        if (measurements.isNotEmpty()) Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text(
                    "Measurements",
                    style = MaterialTheme.typography.titleMedium
                ); measurements.forEach {
                Text(
                    "${it.name}: (${it.centre.x.toInt()}, ${it.centre.y.toInt()})  radius ${
                        "%.1f".format(
                            it.apparentRadiusPx
                        )
                    } px"
                )
            }
            }
        }
        if (resultText.isNotBlank()) Card(Modifier.fillMaxWidth()) {
            Text(
                resultText,
                Modifier.padding(12.dp)
            )
        }
        Spacer(Modifier.height(20.dp))
    }
}

@Composable
private fun MeasurementCanvas(
    image: ImageBitmap,
    measurements: List<ObjectMeasurement>,
    pendingCentre: Point2?,
    onTap: (Point2) -> Unit
) {
    Box(
        Modifier
            .fillMaxWidth()
            .background(Color.Black)
    ) {
        Canvas(
            Modifier
                .fillMaxWidth()
                .aspectRatio(image.width.toFloat() / image.height.toFloat())
                .pointerInput(image) {
                    detectTapGestures { offset ->
                        val sx = image.width.toFloat() / size.width;
                        val sy = image.height.toFloat() / size.height; onTap(
                        Point2(
                            offset.x * sx,
                            offset.y * sy
                        )
                    )
                    }
                }) {
            drawImage(image, dstSize = IntSize(size.width.toInt(), size.height.toInt()))
            val sx = size.width / image.width.toFloat();
            val sy = size.height / image.height.toFloat()
            measurements.forEach { m ->
                val c = Offset(m.centre.x * sx, m.centre.y * sy);
                val r = m.apparentRadiusPx * (sx + sy) / 2f;
                val col = if (m.name == "Jack") Color.Yellow else Color.Green; drawCircle(
                col,
                r,
                c,
                style = Stroke(3f)
            ); drawCircle(col, 5f, c)
            }
            pendingCentre?.let { drawCircle(Color.Red, 7f, Offset(it.x * sx, it.y * sy)) }
        }
    }
}
