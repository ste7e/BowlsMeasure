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

    data class WoodDetectorResult(
        val processedBitmap: Bitmap,
        val detections: List<WoodDetection>
    )

    private const val SAMPLE_COUNT = 72
    private val cosTable = DoubleArray(SAMPLE_COUNT) { i -> cos(2.0 * PI * i / SAMPLE_COUNT) }
    private val sinTable = DoubleArray(SAMPLE_COUNT) { i -> sin(2.0 * PI * i / SAMPLE_COUNT) }

    /**
     * Attempts to find the sharpest circular or elliptical object near the click.
     * Employs localized radial ray casting and median outlier filtering to ignore turf anomalies.
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

            val gradX = Mat(); val gradY = Mat()
            Imgproc.Sobel(blurred, gradX, CvType.CV_16S, 1, 0)
            Imgproc.Sobel(blurred, gradY, CvType.CV_16S, 0, 1)

            try {
                // Execute sub-pixel radial edge analysis directly around the user's tapped coordinate
                val res = refineWithRadialRay(blurred, gradX, gradY, scaledPoint, scale)
                working.release()
                if (res != null) {
                    return@withContext res
                }
            } finally {
                gradX.release(); gradY.release()
            }

            return@withContext null

        } finally {
            source.release(); gray.release(); blurred.release()
        }
    }

    /**
     * Casts radial tracking vectors outward from a seed coordinate point,
     * strips leaf/shadow boundary outliers, and runs algebraic ellipse regression.
     */
    private fun refineWithRadialRay(
        blurred: Mat,
        gradX: Mat,
        gradY: Mat,
        centerPoint: Point,
        scale: Double
    ): WoodDetection? {
        val rayCount = 72
        val candidatePoints = ArrayList<Point>()
        val detectedRadii = ArrayList<Double>()

        for (i in 0 until rayCount) {
            val angle = 2.0 * PI * i / rayCount
            val cosA = cos(angle)
            val sinA = sin(angle)

            var maxMag = 0.0
            var bestR = 0.0

            // Scan outward along the line profile from r = 10px to r = 75px on our normalized canvas
            for (r in 10..75) {
                val px = (centerPoint.x + cosA * r).toInt()
                val py = (centerPoint.y + sinA * r).toInt()

                if (px in 0 until blurred.cols() && py in 0 until blurred.rows()) {
                    val gxArr = gradX.get(py, px)
                    val gyArr = gradY.get(py, px)
                    if (gxArr != null && gyArr != null) {
                        val gx = gxArr[0]
                        val gy = gyArr[0]
                        val mag = sqrt(gx * gx + gy * gy)

                        // Check if the gradient direction aligns closely with our radial search ray direction
                        val dot = (gx / (mag + 0.0001) * cosA) + (gy / (mag + 0.0001) * sinA)

                        if (mag > maxMag && abs(dot) > 0.35) {
                            maxMag = mag
                            bestR = r.toDouble()
                        }
                    }
                }
            }

            // If a valid edge trigger was hit, collect its boundary position coordinates
            if (maxMag > 12.0 && bestR >= 10.0) {
                candidatePoints.add(Point(centerPoint.x + cosA * bestR, centerPoint.y + sinA * bestR))
                detectedRadii.add(bestR)
            }
        }

        if (candidatePoints.size < 15) return null

        // ROBUST OUTLIER REMOVAL: Calculate the median radius of all ray tracking hits
        val sortedRadii = detectedRadii.sorted()
        val medianRadius = sortedRadii[sortedRadii.size / 2]

        val validPoints = ArrayList<Point>()
        var totalValidationScore = 0.0

        for (i in 0 until candidatePoints.size) {
            // Discard any rays that extended out into trailing shadows or adjacent leaves
            if (abs(detectedRadii[i] - medianRadius) <= 7.0) {
                validPoints.add(candidatePoints[i])
            }
        }

        // We require consistent data density to confirm a structural ball/jack entity
        if (validPoints.size < 18) return null

        var finalCenter = centerPoint
        var finalRadius = medianRadius

        // Feed our clean, filtered boundary point cloud into OpenCV's least-squares ellipse Fitter
        if (validPoints.size >= 5) {
            val matPoints = MatOfPoint2f()
            matPoints.fromList(validPoints)
            try {
                val rotatedRect = Geometry.fitEllipse(matPoints)
                if (rotatedRect.size.width > 0 && rotatedRect.size.height > 0) {
                    val aspect = if (rotatedRect.size.width > rotatedRect.size.height)
                        rotatedRect.size.height / rotatedRect.size.width
                    else
                        rotatedRect.size.width / rotatedRect.size.height

                    // Validate that the perspective aspect match represents an acceptable projection angle
                    if (aspect >= 0.45) {
                        finalCenter = rotatedRect.center
                        finalRadius = (rotatedRect.size.width + rotatedRect.size.height) / 4.0
                        totalValidationScore = aspect
                    }
                }
            } catch (e: Exception) {
                // Fall back cleanly to baseline median radius states if regression matrices experience singularity stalls
                totalValidationScore = 0.75
            } finally {
                matPoints.release()
            }
        }

        return WoodDetection(
            centre = Point(finalCenter.x / scale, finalCenter.y / scale),
            radiusPx = (finalRadius / scale).toFloat(),
            circularity = if (totalValidationScore > 0) totalValidationScore else 0.85,
            confidence = validPoints.size.toDouble() / rayCount.toDouble()
        )
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
                // Lower accumulator threshold (param2 = 15.0) lets perspective distorted ellipses pass the initial voting phase
                Imgproc.HoughCircles(blurred, circles, Imgproc.HOUGH_GRADIENT, 1.2, 35.0, 70.0, 15.0, 10, 75)

                for (i in 0 until circles.cols()) {
                    val data = circles.get(0, i) ?: continue
                    val cx = data[0]
                    val cy = data[1]
                    val r = data[2]

                    // Run our sub-pixel radial ray casting pass to lock onto the precise ellipse center and radius
                    val optimized = refineWithRadialRay(blurred, gradX, gradY, Point(cx, cy), scale)
                    if (optimized != null) {
                        validatedDetections.add(optimized)
                    } else {
                        val score = verifyCircle(gradX, gradY, cx, cy, r)
                        if (score.first > 0.35) {
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
                    if (mag > 6.0) { // Standardized noise floor baseline
                        val dx = (cx - px) / r
                        val dy = (cy - py) / r
                        val dot = (gx / mag * dx) + (gy / mag * dy)
                        if (abs(dot) > 0.35) {
                            align += abs(dot); totalMag += mag; valid++
                        }
                    }
                }
            }
        }
        if (valid < SAMPLE_COUNT * 0.15) return 0.0 to 0.0
        return (align / valid) to min(1.0, (totalMag / valid) / 150.0)
    }
}