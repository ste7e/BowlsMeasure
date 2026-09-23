package com.example.bowlsmeasuring

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import java.util.ArrayList
import kotlin.math.*

object RadialGradientAlignmentDetector {

    /**
     * Localized Semi-Auto:
     * 1. Finds a physical mass (dark wood or colored jack) under the tap.
     * 2. Uses Radial Symmetry to snap the circle perfectly to the edges.
     */
    fun detectNear(bitmap: Bitmap, clickedPoint: Point2): BlobDetection? {
        val source = Mat()
        Utils.bitmapToMat(bitmap, source)

        val targetMaxDim = 1024.0
        val scale = if (maxOf(source.cols(), source.rows()) > targetMaxDim)
            targetMaxDim / maxOf(source.cols(), source.rows()) else 1.0

        val scaledPoint = Point(clickedPoint.x * scale, clickedPoint.y * scale)

        // Increase ROI slightly to ensure we see the whole bowl
        val roiSize = 400
        val sx = maxOf(0, (scaledPoint.x - roiSize / 2).toInt()).coerceAtMost(source.cols() - 1)
        val sy = maxOf(0, (scaledPoint.y - roiSize / 2).toInt()).coerceAtMost(source.rows() - 1)
        val w = minOf(roiSize, (source.cols() - sx))
        val h = minOf(roiSize, (source.rows() - sy))

        if (w <= 20 || h <= 20) { source.release(); return null }

        val roiSrc = source.submat(Rect(sx, sy, w, h))
        val hsv = Mat(); val gray = Mat(); val binary = Mat()
        val blurred = Mat(); val gradX = Mat(); val gradY = Mat()

        try {
            Imgproc.cvtColor(roiSrc, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.cvtColor(roiSrc, hsv, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(hsv, hsv, Imgproc.COLOR_RGB2HSV)
            Imgproc.medianBlur(gray, blurred, 5)

            val woodMask = Mat()
            val jackMask = Mat()
            val cyanMask = Mat()
            val yellowMask = Mat()

            // Woods: Tighten the adaptive threshold constant to 10.0 to be less sensitive to broad shadows
            Imgproc.adaptiveThreshold(blurred, woodMask, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 65, 10.0)

            Core.inRange(hsv, Scalar(85.0, 50.0, 50.0), Scalar(105.0, 255.0, 255.0), cyanMask)
            Core.inRange(hsv, Scalar(20.0, 50.0, 50.0), Scalar(40.0, 255.0, 255.0), yellowMask)
            Core.bitwise_or(cyanMask, yellowMask, jackMask)
            Core.bitwise_or(woodMask, jackMask, binary)

            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
            Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel)
            kernel.release()

            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

            val localTapX = scaledPoint.x - sx
            val localTapY = scaledPoint.y - sy
            val localTap = Point(localTapX, localTapY)

            var bestBlob: BlobDetection? = null
            var minDistanceToTap = Double.MAX_VALUE

            Imgproc.Sobel(blurred, gradX, CvType.CV_32F, 1, 0)
            Imgproc.Sobel(blurred, gradY, CvType.CV_32F, 0, 1)

            for (contour in contours) {
                val mop2f = MatOfPoint2f(*contour.toArray())
                val dist = Geometry.pointPolygonTest(mop2f, localTap, true)

                // 1. Must be near the tap
                // 2. Must not be the entire ROI (prevents the 653px error)
                val area = Geometry.contourArea(contour)
                if (dist > -25.0 && area > 100 && area < (w * h * 0.8)) {
                    val moments = Geometry.moments(contour)
                    val cx = moments._m10 / moments._m00
                    val cy = moments._m01 / moments._m00

                    val distanceToCenter = sqrt((cx - localTapX).pow(2) + (cy - localTapY).pow(2))

                    // Only process if this contour center is closer to the tap than the previous best
                    if (distanceToCenter < minDistanceToTap) {
                        val estR = sqrt(area / PI)
                        val refined = refineSymmetry(gradX, gradY, cx, cy, estR)

                        if (refined.score > 0.35) {
                            minDistanceToTap = distanceToCenter
                            bestBlob = BlobDetection(
                                centre = Point2(((refined.cx + sx) / scale).toFloat(), ((refined.cy + sy) / scale).toFloat()),
                                radiusPx = (refined.r / scale).toFloat(),
                                areaPixels = area.toInt(),
                                solidity = refined.score
                            )
                        }
                    }
                }
                mop2f.release()
            }
            return bestBlob

        } finally {
            // ... (cleanup mats)
        }
    }
    private data class Refinement(val cx: Double, val cy: Double, val r: Double, val score: Double)

    private fun refineSymmetry(gradX: Mat, gradY: Mat, initX: Double, initY: Double, initR: Double): Refinement {
        var bestX = initX
        var bestY = initY
        var bestR = initR
        var maxScore = 0.0

        // Small optimization search: nudge center and radius to maximize symmetry
        for (dr in -4..4 step 2) {
            for (dx in -6..6 step 3) {
                for (dy in -6..6 step 3) {
                    val curX = initX + dx
                    val curY = initY + dy
                    val curR = initR + dr
                    val s = calculateSymmetryScore(gradX, gradY, curX, curY, curR)
                    if (s > maxScore) {
                        maxScore = s
                        bestX = curX
                        bestY = curY
                        bestR = curR
                    }
                }
            }
        }
        return Refinement(bestX, bestY, bestR, maxScore)
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
                val gx = gradX.get(py, px)[0]
                val gy = gradY.get(py, px)[0]
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
        if (validPoints < samples * 0.35) return 0.0
        return totalAlignment / validPoints
    }

    fun detect(bitmap: Bitmap): List<BlobDetection> {
        // Implementation for the "Detect Woods" button
        val detections = ArrayList<BlobDetection>()
        val source = Mat()
        Utils.bitmapToMat(bitmap, source)
        val targetMaxDim = 1024.0
        val scale = if (maxOf(source.cols(), source.rows()) > targetMaxDim)
            targetMaxDim / maxOf(source.cols(), source.rows()) else 1.0

        val working = Mat()
        Imgproc.resize(source, working, Size(source.cols() * scale, source.rows() * scale))
        val gray = Mat(); val blurred = Mat(); val gradX = Mat(); val gradY = Mat()
        val circles = Mat()

        try {
            Imgproc.cvtColor(working, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.medianBlur(gray, blurred, 5)
            Imgproc.Sobel(blurred, gradX, CvType.CV_32F, 1, 0)
            Imgproc.Sobel(blurred, gradY, CvType.CV_32F, 0, 1)

            // Low-threshold Hough pass to find candidates
            Imgproc.HoughCircles(blurred, circles, Imgproc.HOUGH_GRADIENT, 1.2, 40.0, 60.0, 18.0, 10, 85)

            for (i in 0 until circles.cols()) {
                val data = circles.get(0, i) ?: continue
                val score = calculateSymmetryScore(gradX, gradY, data[0], data[1], data[2])
                if (score > 0.45) {
                    detections.add(BlobDetection(
                        Point2((data[0] / scale).toFloat(), (data[1] / scale).toFloat()),
                        (data[2] / scale).toFloat(),
                        (PI * data[2] * data[2]).toInt(),
                        score
                    ))
                }
            }
        } finally {
            source.release(); working.release(); gray.release(); blurred.release()
            gradX.release(); gradY.release(); circles.release()
        }
        return detections
    }
}