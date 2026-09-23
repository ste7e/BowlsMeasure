package com.example.bowlsmeasuring

import android.graphics.Bitmap
import androidx.core.graphics.createBitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.geometry.Geometry // NATIVE FOR OPENCV 5.0
import org.opencv.imgproc.Imgproc
import java.util.ArrayList
import kotlin.math.*

data class WoodDetection(
    val centre: Point,
    val radiusPx: Float,
    val circularity: Double,
    val confidence: Double
)

object WoodDetector {

    // Define the result class at the top of the object
    data class WoodDetectorResult(
        val processedBitmap: Bitmap,
        val detections: List<WoodDetection>
    )

    private const val SAMPLE_COUNT = 72
    private val cosTable = DoubleArray(SAMPLE_COUNT) { i -> cos(2.0 * PI * i / SAMPLE_COUNT) }
    private val sinTable = DoubleArray(SAMPLE_COUNT) { i -> sin(2.0 * PI * i / SAMPLE_COUNT) }

    /**
     * Attempts to find the sharpest circular or elliptical object near the click.
     * Uses morphologically stable edge profiles to isolate bowls from lawn noise.
     */
    suspend fun detectNear(bitmap: Bitmap, clickedPoint: Point2): WoodDetection? = withContext(Dispatchers.Default) {
        val source = Mat()
        Utils.bitmapToMat(bitmap, source)

        val targetMaxDim = 1024.0
        val srcW = source.cols().toDouble()
        val srcH = source.rows().toDouble()
        val scale = if (maxOf(srcW, srcH) > targetMaxDim) targetMaxDim / maxOf(srcW, srcH) else 1.0

        val scaledPoint = Point(clickedPoint.x * scale, clickedPoint.y * scale)
        val gray = Mat(); val blurred = Mat()

        try {
            val working = Mat()
            Imgproc.resize(source, working, Size(srcW * scale, srcH * scale))
            Imgproc.cvtColor(working, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.medianBlur(gray, gray, 5)
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 1.5)

            // Primary Target Handler: Localized Contour Arc Regression Analysis
            val res = detectNearContour(blurred, scaledPoint, scale)
            working.release()

            if (res != null) {
                return@withContext res
            }

            // Fallback Strategy: Precise Hough Circle Verification Loop if contour lines are fractured
            val gradX = Mat(); val gradY = Mat()
            Imgproc.Sobel(blurred, gradX, CvType.CV_16S, 1, 0)
            Imgproc.Sobel(blurred, gradY, CvType.CV_16S, 0, 1)

            try {
                val cannySteps = listOf(100.0, 80.0, 60.0)
                val accSteps = listOf(35.0, 22.0, 14.0)

                for (canny in cannySteps) {
                    for (acc in accSteps) {
                        val circles = Mat()
                        try {
                            val dp = if (acc < 20.0) 1.5 else 1.2
                            // Enforced hard limits: minRadius=10, maxRadius=75
                            Imgproc.HoughCircles(blurred, circles, Imgproc.HOUGH_GRADIENT, dp, 30.0, canny, acc, 10, 75)

                            var bestCircle: WoodDetection? = null
                            var minDistance = Double.MAX_VALUE

                            for (i in 0 until circles.cols()) {
                                val data = circles.get(0, i) ?: continue
                                val r = data[2]
                                val dist = hypot(data[0] - scaledPoint.x, data[1] - scaledPoint.y)

                                // Click containment check
                                if (dist <= r * 1.15 && dist < minDistance) {
                                    val score = verifyCircle(gradX, gradY, data[0], data[1], r)
                                    if (score.first > 0.35) {
                                        minDistance = dist
                                        bestCircle = WoodDetection(
                                            centre = Point(data[0] / scale, data[1] / scale),
                                            radiusPx = (r / scale).toFloat(),
                                            circularity = score.first,
                                            confidence = 1.0 - (dist / (r * 1.15))
                                        )
                                    }
                                }
                            }
                            if (bestCircle != null) {
                                return@withContext bestCircle
                            }
                        } finally {
                            circles.release()
                        }
                    }
                }
            } finally {
                gradX.release(); gradY.release()
            }

            return@withContext null

        } finally {
            source.release(); gray.release(); blurred.release()
        }
    }

    private fun detectNearContour(blurred: Mat, scaledPoint: Point, scale: Double): WoodDetection? {
        val roiSize = 240 // Tighter local focus insulates target calculations from background anomalies
        val sx = max(0, (scaledPoint.x - roiSize / 2).toInt()).coerceAtMost(blurred.cols() - 1)
        val sy = max(0, (scaledPoint.y - roiSize / 2).toInt()).coerceAtMost(blurred.rows() - 1)
        val ex = min(blurred.cols() - 1, (scaledPoint.x + roiSize / 2).toInt())
        val ey = min(blurred.rows() - 1, (scaledPoint.y + roiSize / 2).toInt())
        val w = ex - sx; val h = ey - sy
        if (w <= 10 || h <= 10) return null

        val roi = blurred.submat(Rect(sx, sy, w, h))
        val edges = Mat()

        // Localized structural edge extraction ignores global lighting artifacts
        Imgproc.Canny(roi, edges, 40.0, 120.0)

        // Balanced dilation repairs thin tracking boundaries without merging grass segments
        val element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
        Imgproc.dilate(edges, edges, element)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)

        // Compute local gradient maps for geometric validation
        val gradX = Mat(); val gradY = Mat()
        Imgproc.Sobel(roi, gradX, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(roi, gradY, CvType.CV_16S, 0, 1)

        var bestCandidate: WoodDetection? = null
        var maxScore = 0.0

        for (contour in contours) {
            val mop2f = MatOfPoint2f()
            contour.convertTo(mop2f, CvType.CV_32F)

            val areaVal = Geometry.contourArea(mop2f)
            if (areaVal < 80.0) { mop2f.release(); continue }

            val center = Point()
            val radiusArr = floatArrayOf(0f)
            Geometry.minEnclosingCircle(mop2f, center, radiusArr)

            val localX = center.x
            val localY = center.y
            val radius = radiusArr[0].toDouble()

            // CRITICAL CALIBRATION LOCK: Discard giant grass masks and minuscule noise shapes immediately
            if (radius < 10.0 || radius > 75.0) { mop2f.release(); continue }

            val globalCenterX = localX + sx.toDouble()
            val globalCenterY = localY + sy.toDouble()
            val distToClick = hypot(globalCenterX - scaledPoint.x, globalCenterY - scaledPoint.y)

            // Strict Proximity Gate: The finger tap position must land cleanly inside the boundary footprint
            if (distToClick <= radius * 1.15) {
                // Verify if the edge layout possesses circular gradient vectors or random noise configurations
                val score = verifyCircle(gradX, gradY, localX, localY, radius)
                if (score.first > 0.38) {
                    var finalRadius = radius

                    // Extract high-precision perspective ellipse metrics if the boundary footprint is stable
                    if (mop2f.rows() >= 5) {
                        val rotatedRect = Geometry.fitEllipse(mop2f)
                        val rw = rotatedRect.size.width
                        val rh = rotatedRect.size.height
                        if (rw > 0 && rh > 0) {
                            val aspect = if (rw > rh) rh / rw else rw / rh
                            if (aspect >= 0.55) {
                                finalRadius = (rw + rh) / 4.0
                            }
                        }
                    }

                    // Composite scoring balances clean gradient verification and closeness to your tap location
                    val proximityFactor = 1.0 - (distToClick / (radius * 1.5))
                    val compositeScore = score.first * score.second * proximityFactor

                    if (compositeScore > maxScore) {
                        maxScore = compositeScore
                        bestCandidate = WoodDetection(
                            centre = Point(globalCenterX / scale, globalCenterY / scale),
                            radiusPx = (finalRadius / scale).toFloat(),
                            circularity = score.first,
                            confidence = compositeScore
                        )
                    }
                }
            }
            mop2f.release()
        }

        roi.release(); edges.release(); hierarchy.release(); element.release()
        gradX.release(); gradY.release()
        for (c in contours) c.release()
        return bestCandidate
    }

    /**
     * Scans the entire frame using a high-precision multi-scale radial gradient pipeline.
     * Extracts perspective ellipses (woods) and small highlights (jacks) while rejecting turf noise.
     */
    fun detect(bitmap: Bitmap): WoodDetectorResult {
        val source = Mat()
        Utils.bitmapToMat(bitmap, source)

        val targetMaxDim = 1024.0
        val srcW = source.cols().toDouble()
        val srcH = source.rows().toDouble()
        val scale = if (maxOf(srcW, srcH) > targetMaxDim) targetMaxDim / maxOf(srcW, srcH) else 1.0

        val working = Mat()
        Imgproc.resize(source, working, Size(srcW * scale, srcH * scale))

        val gray = Mat()
        val blurred = Mat()

        try {
            Imgproc.cvtColor(working, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.medianBlur(gray, gray, 5)
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 1.5)

            val gradX = Mat(); val gradY = Mat()
            Imgproc.Sobel(blurred, gradX, CvType.CV_16S, 1, 0)
            Imgproc.Sobel(blurred, gradY, CvType.CV_16S, 0, 1)

            val circles = Mat()
            val validatedDetections = ArrayList<WoodDetection>()

            try {
                // Hough limits enforced to physical bowl bounds (10px to 75px)
                Imgproc.HoughCircles(blurred, circles, Imgproc.HOUGH_GRADIENT, 1.2, 40.0, 80.0, 24.0, 10, 75)

                for (i in 0 until circles.cols()) {
                    val data = circles.get(0, i) ?: continue
                    val cx = data[0]
                    val cy = data[1]
                    val r = data[2]

                    // Vet each candidate using our strict gradient vector check
                    val score = verifyCircle(gradX, gradY, cx, cy, r)
                    if (score.first > 0.38) {
                        validatedDetections.add(
                            WoodDetection(
                                centre = Point(cx / scale, cy / scale),
                                radiusPx = (r / scale).toFloat(),
                                circularity = score.first,
                                confidence = score.first * score.second
                            )
                        )
                    }
                }
            } finally {
                circles.release()
                gradX.release()
                gradY.release()
            }

            val cleanDetections = suppressDuplicates(validatedDetections)

            val bitmapResult = createBitmap(blurred.cols(), blurred.rows())
            Utils.matToBitmap(blurred, bitmapResult)

            return WoodDetectorResult(bitmapResult, cleanDetections)

        } finally {
            source.release()
            working.release()
            gray.release()
            blurred.release()
        }
    }

    private fun suppressDuplicates(candidates: List<WoodDetection>): List<WoodDetection> {
        val notNested = candidates.filter { c1 ->
            candidates.none { c2 ->
                if (c1 === c2) return@none false
                val dx = c1.centre.x - c2.centre.x
                val dy = c1.centre.y - c2.centre.y
                val dist = sqrt(dx * dx + dy * dy)
                c1.radiusPx < c2.radiusPx && (dist + c1.radiusPx < c2.radiusPx * 1.05)
            }
        }

        val result = mutableListOf<WoodDetection>()
        for (candidate in notNested.sortedByDescending { it.confidence }) {
            val duplicate = result.any { existing ->
                val dx = existing.centre.x - candidate.centre.x
                val dy = existing.centre.y - candidate.centre.y
                val dist = sqrt(dx * dx + dy * dy)
                dist < minOf(existing.radiusPx, candidate.radiusPx) * 0.7
            }
            if (!duplicate) {
                result += candidate
            }
        }
        return result
    }

    private fun verifyCircle(gradX: Mat, gradY: Mat, cx: Double, cy: Double, r: Double): Pair<Double, Double> {
        var align = 0.0; var totalMag = 0.0; var valid = 0
        for (i in 0 until SAMPLE_COUNT) {
            val px = (cx + cosTable[i] * r).toInt()
            val py = (cy + sinTable[i] * r).toInt()
            if (px in 0 until gradX.cols() && py in 0 until gradX.rows()) {
                val gxArr = gradX.get(py, px); val gyArr = gradY.get(py, px)
                if (gxArr != null && gyArr != null) {
                    val gx = gxArr[0]; val gy = gyArr[0]
                    val mag = sqrt(gx * gx + gy * gy)
                    if (mag > 8.0) { // Discard weak texturing noise floor
                        val dx = (cx - px) / r
                        val dy = (cy - py) / r
                        // Evaluates dot-alignment against theoretical circular radial lines
                        val dot = (gx / mag * dx) + (gy / mag * dy)
                        if (abs(dot) > 0.4) {
                            align += abs(dot); totalMag += mag; valid++
                        }
                    }
                }
            }
        }
        // CRITICAL FILTER: Requires a cohesive arc boundary coverage profile (minimum 20% or 15 aligned markers)
        if (valid < SAMPLE_COUNT * 0.20) return 0.0 to 0.0
        return (align / valid) to min(1.0, (totalMag / valid) / 150.0)
    }
}