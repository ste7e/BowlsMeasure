package com.example.bowlsmeasuring

import android.graphics.Bitmap
import androidx.core.graphics.createBitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.geometry.Geometry // NEW FOR OPENCV 5.0
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

    private const val SAMPLE_COUNT = 72
    private val cosTable = DoubleArray(SAMPLE_COUNT) { i -> cos(2.0 * PI * i / SAMPLE_COUNT) }
    private val sinTable = DoubleArray(SAMPLE_COUNT) { i -> sin(2.0 * PI * i / SAMPLE_COUNT) }

    data class WoodDetectorResult(
        val processedBitmap: Bitmap,
        val detections: List<WoodDetection>
    )

    /**
     * Attempts to find the sharpest circular or elliptical object near the click.
     * Iteratively relaxes Hough thresholds, then falls back to ellipse-fitting.
     */
    suspend fun detectNear(bitmap: Bitmap, clickedPoint: Point2): WoodDetection? = withContext(Dispatchers.Default) {
        val source = Mat()
        Utils.bitmapToMat(bitmap, source)

        val targetMaxDim = 1024.0
        val srcW = source.cols().toDouble()
        val srcH = source.rows().toDouble()
        val scale = if (maxOf(srcW, srcH) > targetMaxDim) targetMaxDim / maxOf(srcW, srcH) else 1.0

        val scaledPoint = Point(clickedPoint.x * scale, clickedPoint.y * scale)
        val gray = Mat(); val blurred = Mat(); val gradX = Mat(); val gradY = Mat()

        try {
            val working = Mat()
            Imgproc.resize(source, working, Size(srcW * scale, srcH * scale))
            Imgproc.cvtColor(working, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.medianBlur(gray, gray, 5)
            Imgproc.GaussianBlur(gray, blurred, Size(7.0, 7.0), 1.5)

            Imgproc.Sobel(blurred, gradX, CvType.CV_16S, 1, 0)
            Imgproc.Sobel(blurred, gradY, CvType.CV_16S, 0, 1)

            // Strategy 1: Hough Circles
            val cannySteps = listOf(110.0, 90.0, 70.0, 50.0)
            val accSteps = listOf(50.0, 35.0, 20.0, 12.0)

            for (canny in cannySteps) {
                for (acc in accSteps) {
                    val circles = Mat()
                    try {
                        val dp = if (acc < 20.0) 1.5 else 1.2
                        // FIX: Reduced maxRadius from 450 to 160. A lawn bowl will never be 450px wide on a 1024px canvas.
                        Imgproc.HoughCircles(blurred, circles, Imgproc.HOUGH_GRADIENT, dp, 30.0, canny, acc, 15, 160)

                        var bestCircle: WoodDetection? = null
                        var maxConfidence = 0.0

                        for (i in 0 until circles.cols()) {
                            val data = circles.get(0, i) ?: continue
                            val r = data[2]
                            val dist = hypot(data[0] - scaledPoint.x, data[1] - scaledPoint.y)

                            // FIX: Tightened click containment threshold from r * 2.2 to r * 1.2.
                            // The click must actually land inside or immediately on the edge of the candidate circle.
                            if (dist < r * 1.2) {
                                val score = verifyCircle(gradX, gradY, data[0], data[1], r)
                                val confidence = score.first * score.second

                                // FIX: Select based on highest physical edge confidence rather than largest radius size
                                if (score.first > 0.35 && confidence > maxConfidence) {
                                    maxConfidence = confidence
                                    bestCircle = WoodDetection(
                                        centre = Point(data[0] / scale, data[1] / scale),
                                        radiusPx = (r / scale).toFloat(),
                                        circularity = score.first,
                                        confidence = confidence
                                    )
                                }
                            }
                        }
                        if (bestCircle != null) {
                            working.release(); circles.release()
                            return@withContext bestCircle
                        }
                    } finally {
                        circles.release()
                    }
                }
            }

            // Strategy 2: Ellipse fitting fallback
            val res = detectNearContour(gray, scaledPoint, scale)
            working.release()
            return@withContext res

        } finally {
            source.release(); gray.release(); blurred.release(); gradX.release(); gradY.release()
        }
    }

    private fun detectNearContour(gray: Mat, scaledPoint: Point, scale: Double): WoodDetection? {
        val roiSize = 650
        val sx = max(0, (scaledPoint.x - roiSize / 2).toInt()).coerceAtMost(gray.cols() - 1)
        val sy = max(0, (scaledPoint.y - roiSize / 2).toInt()).coerceAtMost(gray.rows() - 1)
        val ex = min(gray.cols() - 1, (scaledPoint.x + roiSize / 2).toInt())
        val ey = min(gray.rows() - 1, (scaledPoint.y + roiSize / 2).toInt())
        val w = ex - sx; val h = ey - sy
        if (w <= 10 || h <= 10) return null

        val roi = gray.submat(Rect(sx, sy, w, h))
        val edges = Mat()
        Imgproc.Canny(roi, edges, 25.0, 75.0)
        Imgproc.dilate(edges, edges, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0)))

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        var bestCandidate: WoodDetection? = null
        var maxConfidence = 0.0

        for (contour in contours) {
            val mop2f = MatOfPoint2f()
            contour.convertTo(mop2f, CvType.CV_32F)

            val areaVal = Geometry.contourArea(mop2f)
            if (areaVal < 600.0) { mop2f.release(); continue }

            val center = Point()
            val radiusArr = floatArrayOf(0f)
            Geometry.minEnclosingCircle(mop2f, center, radiusArr)

            val globalCenterX = center.x + sx.toDouble()
            val globalCenterY = center.y + sy.toDouble()
            val distToClick = hypot(globalCenterX - scaledPoint.x, globalCenterY - scaledPoint.y)

            // FIX: Tightened contour neighborhood verification to r * 1.2
            if (distToClick < radiusArr[0].toDouble() * 1.2) {
                val effectiveRadius = if (mop2f.rows() >= 5) {
                    val rotatedRect = Geometry.fitEllipse(mop2f)
                    (rotatedRect.size.width + rotatedRect.size.height) / 4.0
                } else radiusArr[0].toDouble()

                val circularity = areaVal / (PI * radiusArr[0].toDouble() * radiusArr[0].toDouble())

                // FIX: Select based on best geometric circularity match rather than largest total canvas area
                if (circularity > 0.40 && circularity > maxConfidence) {
                    maxConfidence = circularity
                    bestCandidate = WoodDetection(
                        centre = Point(globalCenterX / scale, globalCenterY / scale),
                        radiusPx = (effectiveRadius / scale).toFloat(),
                        circularity = circularity,
                        confidence = 0.6
                    )
                }
            }
            mop2f.release()
        }

        roi.release(); edges.release(); hierarchy.release()
        for (c in contours) c.release()
        return bestCandidate
    }

    suspend fun detect(bitmap: Bitmap): WoodDetectorResult = withContext(Dispatchers.Default) {
        val source = Mat()
        Utils.bitmapToMat(bitmap, source)
        val targetMaxDim = 1024.0
        val srcW = source.cols().toDouble(); val srcH = source.rows().toDouble()
        val scale = if (maxOf(srcW, srcH) > targetMaxDim) targetMaxDim / maxOf(srcW, srcH) else 1.0
        Imgproc.resize(source, source, Size(srcW * scale, srcH * scale))
        val gray = Mat(); val blurred = Mat()
        Imgproc.cvtColor(source, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.medianBlur(gray, gray, 5)
        Imgproc.GaussianBlur(gray, blurred, Size(7.0, 7.0), 1.5)
        val lowRes = ArrayList<WoodDetection>()
        val circles = Mat()
        // FIX: Also capped global detector max radius to 160
        Imgproc.HoughCircles(blurred, circles, Imgproc.HOUGH_GRADIENT, 1.2, 50.0, 90.0, 45.0, 20, 160)
        for (i in 0 until circles.cols()) {
            val d = circles.get(0, i) ?: continue
            lowRes.add(WoodDetection(Point(d[0] / scale, d[1] / scale), (d[2] / scale).toFloat(), 1.0, 1.0))
        }
        val bitmapResult = createBitmap(blurred.cols(), blurred.rows())
        Utils.matToBitmap(blurred, bitmapResult)

        gray.release(); blurred.release(); circles.release(); source.release()
        WoodDetectorResult(bitmapResult, lowRes)
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
                    if (mag > 5.0) {
                        val dot = (gx / mag * ((cx - px) / r)) + (gy / mag * ((cy - py) / r))
                        if (abs(dot) > 0.3) {
                            align += abs(dot); totalMag += mag; valid++
                        }
                    }
                }
            }
        }
        return if (valid < SAMPLE_COUNT * 0.08) 0.0 to 0.0 else (align / valid) to min(1.0, (totalMag / valid) / 300.0)
    }
}