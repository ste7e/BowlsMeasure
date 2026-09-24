package com.example.bowlsmeasuring

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import java.util.ArrayList
import java.util.Locale
import kotlin.math.*

/**
 * Debug-enabled WoodDetector with intermediate image logging and numerical stats.
 * Produces debug images at each processing stage to pinpoint where detections fail.
 */
object WoodDetector {
    /**
     * Debug result structure for each candidate
     */
    data class CandidateDebugInfo(
        val centre: Point2,
        val initialRadiusPx: Float,
        val radialPoints: List<Point>,
        val coverageScore: Double,
        val aspectRatio: Double,
        val radialConsistency: Double,
        val validationScore: Double,
        val decision: String // "accepted" or "rejected" and reason
    )

    /**
     * Intermediate processing stage result
     */
    data class ProcessingStage(
        val name: String,
        val bitmap: Bitmap?,
        val numericalStats: Map<String, Double>
    )

    /**
     * Standard detect function
     */
    fun detect(bitmap: Bitmap): List<WoodDetection> {
        return detectWithDebug(bitmap)?.finalDetections ?: emptyList()
    }

    /**
     * Standard detectNear function
     */
    fun detectNear(bitmap: Bitmap, clickedPoint: Point2): WoodDetection? {
        return detectNearWithDebug(bitmap, clickedPoint)?.finalDetections?.firstOrNull()
    }

    /**
     * Main detect function with debug output
     */
    fun detectWithDebug(bitmap: Bitmap): DebugResult? {
        return performDetection(bitmap, null)
    }

    /**
     * Localized detect function with debug output, focused near a clicked point.
     */
    fun detectNearWithDebug(bitmap: Bitmap, clickedPoint: Point2): DebugResult? {
        return performDetection(bitmap, clickedPoint)
    }

    private fun performDetection(bitmap: Bitmap, clickedPoint: Point2?): DebugResult? {
        val debugImages = mutableListOf<Bitmap>()
        val stages = mutableListOf<ProcessingStage>()
        val candidates = mutableListOf<CandidateDebugInfo>()

        println("\nWoodDetector${if (clickedPoint != null) " (Near click)" else ""}")
        println("----------------------------")
        println("Image: ${bitmap.width} x ${bitmap.height}")
        if (clickedPoint != null) {
            println("Click: (${clickedPoint.x.toInt()}, ${clickedPoint.y.toInt()})")
        }

        val source = Mat()
        val working = Mat()
        val gray = Mat()
        val blurred = Mat()
        val heavyBlurred = Mat()
        val gradX = Mat()
        val gradY = Mat()
        val circles = Mat()
        val hsv = Mat()
        val cyanMask = Mat()
        val yellowMask = Mat()
        val jackMask = Mat()
        val woodMask = Mat()
        val binary = Mat()

        return try {
            // Stage 01: Original image
            val originalCopy = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)
            debugImages.add(originalCopy)

            Utils.bitmapToMat(bitmap, source)

            val srcW = source.cols().toDouble()
            val srcH = source.rows().toDouble()
            val scale = if (maxOf(srcW, srcH) > 1024.0) 1024.0 / maxOf(srcW, srcH) else 1.0

            Imgproc.resize(source, working, Size(srcW * scale, srcH * scale))
            Imgproc.cvtColor(working, gray, Imgproc.COLOR_RGBA2GRAY)

            println("Working image: ${working.cols()} x ${working.rows()}")
            logMatStats(working, "Working ROI", debugImages, stages)

            // ✅ NEW: Create HSV color mask to isolate jacks
            val hsv = Mat()
            Core.inRange(
                working,
                Scalar(85.0, 50.0, 50.0),
                Scalar(105.0, 255.0, 255.0),
                cyanMask
            ) // Blue jack
            Core.inRange(
                working,
                Scalar(20.0, 50.0, 50.0),
                Scalar(40.0, 255.0, 255.0),
                yellowMask
            ) // Yellow jacks
            Core.bitwise_or(cyanMask, yellowMask, jackMask)

            Imgproc.medianBlur(gray, blurred, 5)

            // ✅ NEW: Combine wood mask with jack mask for final detection
            Imgproc.adaptiveThreshold(
                blurred, woodMask, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV, 65, 15.0
            )
            Core.bitwise_or(woodMask, jackMask, binary) // OR jacks + woods

            Imgproc.GaussianBlur(blurred, blurred, Size(5.0, 5.0), 1.5)
            Imgproc.medianBlur(gray, blurred, 5)
            Imgproc.GaussianBlur(blurred, blurred, Size(5.0, 5.0), 1.5)

            // Stage 02: Blurred
            logMatStats(blurred, "Blurred", debugImages, stages)
            printStageStats(stages, "Blurred", mapOf("sigma" to 12), "Blur")

            // Stage 03: Heavy blur
            Imgproc.GaussianBlur(blurred, heavyBlurred, Size(13.0, 13.0), 1.5)
            logMatStats(heavyBlurred, "Heavy Blur", debugImages, stages)

            Imgproc.Sobel(blurred, gradX, CvType.CV_16S, 1, 0)
            Imgproc.Sobel(blurred, gradY, CvType.CV_16S, 0, 1)

            // Candidate regions detection (Contours)
            val binary = Mat()
            Imgproc.adaptiveThreshold(
                heavyBlurred,
                binary,
                255.0,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY_INV,
                65,
                10.0
            )
            val allContours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(
                binary,
                allContours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
            )
            val filteredContours = allContours.filter {
                val area = Geometry.contourArea(it)
                area > 200 && area < (working.cols() * working.rows() * 0.25)
            }

            // HoughCircles for additional candidates
            Imgproc.HoughCircles(
                heavyBlurred,
                circles,
                Imgproc.HOUGH_GRADIENT,
                1.2,
                40.0,
                80.0,
                24.0,
                10,
                75
            )

            logMatStats(gradX, "Grad X", debugImages, stages)
            logMatStats(gradY, "Grad Y", debugImages, stages)
            logMatStats(circles, "Circles", debugImages, stages)

            val circlesStats = stages.last().numericalStats
            println("\nCandidate response:")
            println(String.format(Locale.US, "  min = %.1f", circlesStats["min"] ?: 0.0))
            println(String.format(Locale.US, "  max = %.1f", circlesStats["max"] ?: 0.0))
            println(String.format(Locale.US, "  mean = %.1f", circlesStats["mean"] ?: 0.0))
            println("  threshold = 24.0")

            // Merge candidates from Hough and Contours
            val candidatesToTest = mutableListOf<Triple<Double, Double, Double>>() // cx, cy, r
            for (i in 0 until circles.cols()) {
                val data = circles.get(0, i) ?: continue

                // Avoid overlapping candidates
                if (candidatesToTest.any {
                        sqrt(
                            (it.first - data[0]).pow(2) + (it.second - data[1]).pow(
                                2
                            )
                        ) < it.third * 0.7
                    }) {
                    continue
                }

                candidatesToTest.add(Triple(data[0], data[1], data[2]))
            }
            for (contour in filteredContours) {
                val moments = Geometry.moments(contour)
                if (moments._m00 > 0) {
                    val cx = moments._m10 / moments._m00
                    val cy = moments._m01 / moments._m00
                    val area = moments._m00
                    val r = sqrt(area / PI)
                    // Avoid overlapping candidates
                    if (candidatesToTest.none { sqrt((it.first - cx).pow(2) + (it.second - cy).pow(2)) < it.third * 0.7 }) {
                        candidatesToTest.add(Triple(cx, cy, r))
                    }
                }
            }

            // If local mode, filter by distance to click or fallback to click itself
            val finalCandidateCentres = if (clickedPoint != null) {
                val scaledClick = Point(clickedPoint.x * scale, clickedPoint.y * scale)
                val near = candidatesToTest.filter {
                    sqrt((it.first - scaledClick.x).pow(2) + (it.second - scaledClick.y).pow(2)) < 120.0 * scale
                }
                near.ifEmpty {
                    listOf(Triple(scaledClick.x, scaledClick.y, 40.0 * scale))
                }
            } else {
                candidatesToTest
            }

            println("\nCandidate regions:")
            println("  contours = ${allContours.size}")
            println("  after area filter = ${filteredContours.size}")
            println("  candidate centres = ${finalCandidateCentres.size}")

            for (i in finalCandidateCentres.indices) {
                val (cx, cy, r) = finalCandidateCentres[i]

                val score = calculateSymmetryScore(gradX, gradY, cx, cy, r)

                // Collect radial edge points for consistency checks
                val radialPoints = collectRadialPoints(heavyBlurred, gradX, gradY, Point(cx, cy), r)

                // Refinement logic: If we found enough radial points, update the centre and radius
                var finalCx = cx
                var finalCy = cy
                var finalR = r
                if (radialPoints.size >= 10) {
                    finalCx = radialPoints.map { it.x }.average()
                    finalCy = radialPoints.map { it.y }.average()
                    finalR =
                        radialPoints.map { sqrt((it.x - finalCx).pow(2) + (it.y - finalCy).pow(2)) }
                            .average()
                }

                val scaledCentre = Point2((finalCx / scale).toFloat(), (finalCy / scale).toFloat())
                val finalRadius = (finalR / scale).toFloat()

                val consistency = calculateRadialConsistency(radialPoints, Point(finalCx, finalCy))
                val aspectRatio = calculateAspectRatio(radialPoints, Point(finalCx, finalCy))
                val coverage = calculateCoverage(radialPoints)

                val likelyShadow = rejectOnBrightness(
                    doubleArrayOf(finalCx, finalCy, finalR),
                    gray,
                    scale,
                    consistency
                )

                val decision =
                    if (score > 0.45 && consistency > 0.7 && !likelyShadow) "accepted" else "rejected"

                val cand = CandidateDebugInfo(
                    centre = scaledCentre,
                    initialRadiusPx = finalRadius,
                    radialPoints = radialPoints,
                    coverageScore = coverage,
                    aspectRatio = aspectRatio,
                    radialConsistency = consistency,
                    validationScore = score,
                    decision = decision
                )
                candidates.add(cand)

                println(String.format(Locale.US, "\nCandidate %d:", i))
                println(
                    String.format(
                        Locale.US,
                        "  centre = (%d, %d)",
                        finalCx.toInt(),
                        finalCy.toInt()
                    )
                )
                println(String.format(Locale.US, "  initial radius = %d", finalR.toInt()))
                println(String.format(Locale.US, "  radial points = %d", radialPoints.size))
                println(String.format(Locale.US, "  ellipse coverage = %.2f", coverage))
                println(String.format(Locale.US, "  aspect ratio = %.2f", aspectRatio))
                println(String.format(Locale.US, "  radial consistency = %.2f", consistency))
                println("  FINAL = $decision")
            }

            val finalDetections = ArrayList<WoodDetection>()
            for (cand in candidates) {
                if (cand.decision == "accepted") {
                    finalDetections.add(
                        WoodDetection(
                            centre = Point(cand.centre.x.toDouble(), cand.centre.y.toDouble()),
                            radiusPx = cand.initialRadiusPx,
                            circularity = cand.radialConsistency,
                            confidence = cand.validationScore
                        )
                    )
                }
            }

            println("\nFinal woods = ${finalDetections.size}")

            val finalDebugBmp = buildDebugImage(bitmap, candidates)
            debugImages.add(finalDebugBmp)

            DebugResult(
                debugImages = debugImages,
                stages = stages,
                candidates = candidates,
                finalDetections = finalDetections
            )

        } catch (e: Exception) {
            e.printStackTrace()
            null
        } finally {
            source.release(); working.release(); gray.release(); blurred.release()
            heavyBlurred.release(); gradX.release(); gradY.release()
            circles.release()

            hsv.release(); cyanMask.release(); yellowMask.release(); jackMask.release() // ✅ NEW
            woodMask.release(); binary.release() // ✅ NEW
        }
    }


    private fun rejectOnBrightness(data: DoubleArray, gray: Mat, scale: Double, circularity: Double): Boolean {
        val cx = data[0]
        val cy = data[1]
        val r = data[2]

        // ✅ NEW: Check brightness at candidate center to reject shadow-dominant regions
        val innerX = (cx / scale).toInt().coerceIn(0, gray.cols() - 1)
        val innerY = (cy / scale).toInt().coerceIn(0, gray.rows() - 1)

        // Shadows are dark (< 80 in grayscale), blue jacks are bright (> 120)
        val brightness = gray.get(innerY, innerX)?.firstOrNull() ?: 255.0

        // Only accept candidates that are reasonably bright (not shadows)
        if (brightness < 120 && circularity > 0.95) return true  // Skip shadow-dominant candidates

        return false
    }
    private fun printStageStats(stages: List<ProcessingStage>, stageName: String, extraInfo: Map<String, Any> = emptyMap(), displayName: String = stageName) {
        val stage = stages.find { it.name == stageName } ?: return
        println("\n$displayName:")
        extraInfo.forEach { (k, v) -> println("  $k = $v") }
        val stats = stage.numericalStats
        println(String.format(Locale.US, "  min = %.1f", stats["min"] ?: 0.0))
        println(String.format(Locale.US, "  max = %.1f", stats["max"] ?: 0.0))
        println(String.format(Locale.US, "  mean = %.1f", stats["mean"] ?: 0.0))
    }

    private fun logMatStats(mat: Mat, prefix: String, debugImages: MutableList<Bitmap>, stages: MutableList<ProcessingStage>) {
        if (mat.empty()) return

        // 1. Numerical Stats Calculation Safely
        var minVal = 0.0
        var maxVal = 0.0
        try {
            if (mat.channels() == 1) {
                val mm = Core.minMaxLoc(mat)
                minVal = mm.minVal
                maxVal = mm.maxVal
            } else {
                val reshaped = mat.reshape(1)
                val mm = Core.minMaxLoc(reshaped)
                minVal = mm.minVal
                maxVal = mm.maxVal
                reshaped.release()
            }
        } catch (_: Exception) { }
        val meanVal = Core.mean(mat).`val`[0]

        // 2. Safe Bitmap Creation (Only for multidimensional image layouts)
        val isImage = mat.rows() > 1 && mat.cols() > 1 && mat.total() < 10000000
        var bmpResult: Bitmap? = null

        if (isImage) {
            val visualMat = Mat()
            try {
                // Utils.matToBitmap strictly requires CV_8U depth
                when (mat.depth()) {
                    CvType.CV_8U -> mat.copyTo(visualMat)
                    CvType.CV_16S -> Core.convertScaleAbs(mat, visualMat) // Correct for Sobel gradients
                    else -> Core.normalize(mat, visualMat, 0.0, 255.0, Core.NORM_MINMAX, CvType.CV_8U)
                }

                // Supported channel counts for matToBitmap are 1, 3, or 4
                if (visualMat.channels() == 2) {
                    val grayMat = Mat()
                    Imgproc.cvtColor(visualMat, grayMat, Imgproc.COLOR_BGR2GRAY)
                    grayMat.copyTo(visualMat)
                    grayMat.release()
                }

                val bmp = Bitmap.createBitmap(visualMat.cols(), visualMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(visualMat, bmp)
                debugImages.add(bmp)
                bmpResult = bmp
            } catch (_: Exception) {
                bmpResult = null
            } finally {
                visualMat.release()
            }
        }

        stages.add(ProcessingStage(
            name = prefix,
            bitmap = bmpResult,
            numericalStats = mapOf("min" to minVal, "max" to maxVal, "mean" to meanVal)
        ))
    }

    private fun calculateSymmetryScore(gradX: Mat, gradY: Mat, cx: Double, cy: Double, r: Double): Double {
        var totalAlignment = 0.0
        var validPoints = 0
        val samples = 36

        for (i in 0 until samples) {
            val angle = (i * 10.0) * PI / 180.0
            val px = (cx + cos(angle) * r).toInt()
            val py = (cy + sin(angle) * r).toInt()

            if (px >= 0 && px < gradX.cols() && py >= 0 && py < gradX.rows()) {
                val gxArr = gradX.get(py, px)
                val gyArr = gradY.get(py, px)
                if (gxArr != null && gyArr != null) {
                    val gx = gxArr[0]
                    val gy = gyArr[0]
                    val mag = sqrt(gx * gx + gy * gy)

                    if (mag > 12.0) {
                        val nx = gx / mag
                        val ny = gy / mag
                        val rx = cos(angle)
                        val ry = sin(angle)

                        val alignment = abs(nx * rx + ny * ry)
                        if (alignment > 0.75) {
                            totalAlignment += alignment
                            validPoints++
                        }
                    }
                }
            }
        }
        if (validPoints < samples * 0.35) return 0.0
        return totalAlignment / validPoints
    }

    private fun collectRadialPoints(blurred: Mat, gradX: Mat, gradY: Mat, centre: Point, radius: Double): List<Point> {
        val points = mutableListOf<Point>()
        val rayCount = 72
        for (i in 0 until rayCount) {
            val angle = 2.0 * PI * i / rayCount
            val cosA = cos(angle)
            val sinA = sin(angle)

            var maxMag = 0.0
            var bestR = 0.0

            for (r in (radius * 0.7).toInt()..(radius * 1.3).toInt()) {
                val px = (centre.x + cosA * r).toInt()
                val py = (centre.y + sinA * r).toInt()

                if (px in 0 until blurred.cols() && py in 0 until blurred.rows()) {
                    val gxArr = gradX.get(py, px)
                    val gyArr = gradY.get(py, px)
                    if (gxArr != null && gyArr != null) {
                        val gx = gxArr[0]
                        val gy = gyArr[0]
                        val mag = sqrt(gx * gx + gy * gy)
                        val dot = (gx / (mag + 0.0001) * cosA) + (gy / (mag + 0.0001) * sinA)

                        if (mag > maxMag && abs(dot) > 0.35) {
                            maxMag = mag
                            bestR = r.toDouble()
                        }
                    }
                }
            }
            if (maxMag > 12.0 && bestR > 0) {
                points.add(Point(centre.x + cosA * bestR, centre.y + sinA * bestR))
            }
        }
        return points
    }

    private fun calculateRadialConsistency(points: List<Point>, centre: Point): Double {
        if (points.isEmpty()) return 0.0
        val radii = points.map { p -> sqrt((p.x - centre.x).pow(2) + (p.y - centre.y).pow(2)) }
        val meanR = radii.average()
        if (meanR < 1e-6) return 0.0
        val variance = radii.map { (it - meanR).pow(2) }.average()
        return 1.0 / (1.0 + sqrt(variance) / meanR)
    }

    private fun calculateAspectRatio(points: List<Point>, centre: Point): Double {
        if (points.size < 5) return 1.0
        val dx = points.map { it.x - centre.x }
        val dy = points.map { it.y - centre.y }
        val varX = dx.map { it * it }.average()
        val varY = dy.map { it * it }.average()
        return if (varY > 1e-6) sqrt(varX / varY) else 1.0
    }

    private fun calculateCoverage(points: List<Point>): Double {
        if (points.isEmpty()) return 0.0
        return points.size.toDouble() / 72.0
    }

    private fun buildDebugImage(bitmap: Bitmap, candidates: List<CandidateDebugInfo>): Bitmap {
        val debug = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(debug)
        val paint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 3f }
        val textPaint = Paint().apply { color = Color.WHITE; textSize = 24f }

        for ((index, cand) in candidates.withIndex()) {
            paint.color = if (cand.decision.startsWith("accepted")) Color.GREEN else Color.RED
            canvas.drawCircle(cand.centre.x, cand.centre.y, cand.initialRadiusPx, paint)

            val text = "Cand $index: ${cand.decision}"
            canvas.drawText(text, cand.centre.x - 50f, cand.centre.y - cand.initialRadiusPx - 10f, textPaint)
        }
        return debug
    }

    /**
     * Debug result structure
     */
    data class DebugResult(
        val debugImages: List<Bitmap>,
        val stages: List<ProcessingStage>,
        val candidates: List<CandidateDebugInfo>,
        val finalDetections: List<WoodDetection>
    )
}
