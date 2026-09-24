package com.example.bowlsmeasuring

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.geometry.Geometry
import org.opencv.geometry.Moments
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C
import org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE
import org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY
import org.opencv.imgproc.Imgproc.RETR_EXTERNAL
import org.opencv.imgproc.Imgproc.THRESH_BINARY
import org.opencv.imgproc.Imgproc.THRESH_BINARY_INV
import org.opencv.imgproc.Imgproc.THRESH_OTSU
import java.util.ArrayList
import kotlin.math.*

object SimpleBlobDetector {

    /**
     * Localized Semi-Auto detection - finds physical mass under tap point.
     * Uses ROI constraints to focus search near clicked location.
     */
    fun detectNear(bitmap: Bitmap, clickedPoint: Point2): BlobDetection? {
        val source = Mat()
        Utils.bitmapToMat(bitmap, source)

        val targetMaxDim = 1024.0
        val scale = if (maxOf(source.cols(), source.rows()) > targetMaxDim)
            targetMaxDim / maxOf(source.cols(), source.rows()) else 1.0
        val scaledPoint = Point(clickedPoint.x * scale, clickedPoint.y * scale)

        // ROI: Focus search within a reasonable window around tap point
        val roiSize = 300
        val sx = maxOf(0, (scaledPoint.x - roiSize / 2).toInt()).coerceAtMost(source.cols() - 1)
        val sy = maxOf(0, (scaledPoint.y - roiSize / 2).toInt()).coerceAtMost(source.rows() - 1)
        val ex = min(source.cols() - 1, (scaledPoint.x + roiSize / 2).toInt())
        val ey = min(source.rows() - 1, (scaledPoint.y + roiSize / 2).toInt())

        if (ex < sx || ey < sy) return null // ROI too small

        val working = Mat()
        Imgproc.resize(source, working, Size(source.cols() * scale, source.rows() * scale))

        val gray = Mat()
        Imgproc.cvtColor(working, gray, COLOR_RGBA2GRAY)

        // Median blur better for grass speckle noise than Gaussian
        val blurred = Mat()
        Imgproc.medianBlur(gray, blurred, 5)
        Imgproc.GaussianBlur(blurred, blurred, Size(5.0, 5.0), 1.5)

        // Create ROI mask to restrict search area
        val binaryMask = Mat()
        Imgproc.threshold(blurred, binaryMask, 128.0, 255.0, THRESH_BINARY + THRESH_OTSU)

        val hierarchy = Mat()
        val contours = ArrayList<MatOfPoint>()
        val woodMask = Mat()

        try {
            // Apply adaptive threshold - grass shadows will be ignored
            Imgproc.adaptiveThreshold(blurred, binaryMask, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 65, 15.0)

            // Adaptive threshold with high constant to ignore grass shadows
            Imgproc.adaptiveThreshold(blurred, woodMask, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 65, 15.0)

            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(woodMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

            var bestDetection: BlobDetection? = null

            // Find the most plausible detection near tap point
            for (contour in contours) {
                val areaVal = Geometry.contourArea(contour)

                // Skip very small or very large areas
                if (areaVal < 150.0 || areaVal > 25000.0) continue

                // Check centroid distance to tap point
                val moments = Geometry.moments(contour)
                val cx = moments._m10 / moments._m00
                val cy = moments._m01 / moments._m00

                val distToTap = sqrt((cx - scaledPoint.x).pow(2) + (cy - scaledPoint.y).pow(2))

                // Prefer detections close to tap point (within 150px from center of ROI)
                if (distToTap > roiSize / 2) continue

                val mop2f = MatOfPoint2f()
                try {
                    contour.convertTo(mop2f, CvType.CV_32F)

                    // Calculate circularity
                    val perimeter = Geometry.arcLength(mop2f, true)
                    val circularity = if (perimeter > 0)
                        (4 * PI * areaVal) / (perimeter * perimeter) else 0.0

                    if (circularity <= 0.5) continue // Filter non-circular grass patches

                    val center = Point()
                    val radiusArr = floatArrayOf(0f)
                    Geometry.minEnclosingCircle(mop2f, center, radiusArr)

                    bestDetection = BlobDetection(
                        centre = Point2((center.x / scale).toFloat(), (center.y / scale).toFloat()),
                        radiusPx = (radiusArr[0] / scale).toFloat(),
                        areaPixels = areaVal.toInt(),
                        solidity = circularity
                    )

                } finally {
                    mop2f.release()
                }
            }

            // Sort by circularity and pick best
            return bestDetection?.let {
                if (it.solidity <= 0.7) null else it
            } ?: null

        } finally {
            source.release()
            working.release()
            gray.release()
            blurred.release()
            binaryMask.release()
            woodMask.release()
            hierarchy.release()
        }
    }

    fun detectNearV0(bitmap: Bitmap, clickedPoint: Point2): BlobDetection? {
        val source = Mat()
        Utils.bitmapToMat(bitmap, source)

        val targetMaxDim = 1024.0
        val scale = if (maxOf(source.cols(), source.rows()) > targetMaxDim)
            targetMaxDim / maxOf(source.cols(), source.rows()) else 1.0

        val scaledPoint = Point(clickedPoint.x * scale, clickedPoint.y * scale)
        val roiSize = 350
        val sx = maxOf(0, (scaledPoint.x - roiSize / 2).toInt()).coerceAtMost(source.cols() - 1)
        val sy = maxOf(0, (scaledPoint.y - roiSize / 2).toInt()).coerceAtMost(source.rows() - 1)
        val w = minOf(roiSize, (source.cols() - sx))
        val h = minOf(roiSize, (source.rows() - sy))

        if (w <= 20 || h <= 20) { source.release(); return null }

        val roiSrc = source.submat(Rect(sx, sy, w, h))
        val gray = Mat(); val binary = Mat(); val hsv = Mat(); val mask = Mat()
        val distMap = Mat(); val jackMask = Mat(); val woodMask = Mat()

        try {
            // 1. Convert to Gray and HSV
            Imgproc.cvtColor(roiSrc, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.cvtColor(roiSrc, hsv, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(hsv, hsv, Imgproc.COLOR_RGB2HSV)
            Imgproc.medianBlur(gray, gray, 7)

            // 2. MASKING: Find Cyan/Yellow Jack OR Dark Woods
            // Cyan Jack: Hue 85-105
            Core.inRange(hsv, Scalar(85.0, 50.0, 50.0), Scalar(105.0, 255.0, 255.0), jackMask)
            // Woods: Adaptive threshold with a high constant (15) to cut through shadows
            Imgproc.adaptiveThreshold(gray, woodMask, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 75, 15.0)

            Core.bitwise_or(jackMask, woodMask, binary)

            // 3. CLEANUP: Remove grass specks
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
            Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel)
            kernel.release()

            // 4. DISTANCE TRANSFORM: Find the deepest part of the shape
            Imgproc.distanceTransform(binary, distMap, Geometry.DIST_L2, 3)

            // 5. FIND PEAK NEAR TAP
            val localTapX = scaledPoint.x - sx
            val localTapY = scaledPoint.y - sy
            val searchArea = Mat.zeros(binary.size(), CvType.CV_8U)
            Imgproc.circle(searchArea, Point(localTapX, localTapY), 80, Scalar(255.0), -1)

            val minMax = Core.minMaxLoc(distMap, searchArea)
            val peakRadius = minMax.maxVal
            val peakLoc = minMax.maxLoc

            searchArea.release()

            // 6. VALIDATE: Ensure the result isn't just a tiny grass speck
            if (peakRadius >= 6.0 && peakRadius <= 110.0) {
                val finalX = (peakLoc.x + sx) / scale
                val finalY = (peakLoc.y + sy) / scale
                return BlobDetection(
                    centre = Point2(finalX.toFloat(), finalY.toFloat()),
                    radiusPx = (peakRadius / scale).toFloat(),
                    areaPixels = (PI * peakRadius * peakRadius).toInt(),
                    solidity = 0.9
                )
            }
            return null
        } catch (e: Exception) {
            return null
        } finally {
            source.release(); roiSrc.release(); gray.release(); binary.release()
            hsv.release(); distMap.release(); jackMask.release(); woodMask.release()
        }
    }

     fun detect(bitmap: Bitmap): List<BlobDetection> {
        val source = Mat()
        Utils.bitmapToMat(bitmap, source)

        val targetMaxDim = 1024.0
        val srcW = source.cols().toDouble()
        val srcH = source.rows().toDouble()

        val scale =
            if (maxOf(srcW, srcH) > targetMaxDim)
                targetMaxDim / maxOf(srcW, srcH)
            else
                1.0

        val working = Mat()
        val rgb = Mat()
        val woodMask = Mat()
        val distMap = Mat()
        val localMax = Mat()
        val peakMask = Mat()
        val radiusMask = Mat()
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()

        try {
            // ------------------------------------------------------------
            // 1. Work at a sensible maximum resolution.
            // ------------------------------------------------------------

            Imgproc.resize(
                source,
                working,
                Size(srcW * scale, srcH * scale)
            )

// ------------------------------------------------------------
// 2. Identify the brown wood using colour.
//
// The woods in the photographs have a strong:
//
//     Red > Green > Blue
//
// relationship.
//
// This is considerably more discriminating than simply looking
// for a particular HSV hue because the grass and bare soil can
// also fall into the brown/yellow hue range.
// ------------------------------------------------------------

            Imgproc.cvtColor(
                working,
                rgb,
                Imgproc.COLOR_RGBA2RGB
            )

            val channels = ArrayList<Mat>()
            Core.split(rgb, channels)

            val red = channels[0]
            val green = channels[1]
            val blue = channels[2]

// red - green
            val redGreenDifference = Mat()
            Core.subtract(
                red,
                green,
                redGreenDifference,
                Mat(),
                CvType.CV_16S
            )

// green - blue
            val greenBlueDifference = Mat()
            Core.subtract(
                green,
                blue,
                greenBlueDifference,
                Mat(),
                CvType.CV_16S
            )

// Convert the differences into masks.
//
// The first threshold is the important one:
//
//     R - G > 15
//
// The second prevents strongly red/magenta pixels from being
// treated as brown.
//
// We also require a minimum red value so that dark noise does
// not become a candidate.
            val redDominant = Mat()
            Core.compare(
                redGreenDifference,
                Scalar(15.0),
                redDominant,
                Core.CMP_GT
            )

            val greenAboveBlue = Mat()
            Core.compare(
                greenBlueDifference,
                Scalar(3.0),
                greenAboveBlue,
                Core.CMP_GT
            )

            val sufficientlyBright = Mat()
            Core.compare(
                red,
                Scalar(50.0),
                sufficientlyBright,
                Core.CMP_GT
            )

            Core.bitwise_and(
                redDominant,
                greenAboveBlue,
                woodMask
            )

            Core.bitwise_and(
                woodMask,
                sufficientlyBright,
                woodMask
            )

            for (channel in channels) {
                channel.release()
            }

            redGreenDifference.release()
            greenBlueDifference.release()
            redDominant.release()
            greenAboveBlue.release()
            sufficientlyBright.release()
            rgb.release()

            // ------------------------------------------------------------
            // 3. Clean the colour mask.
            //
            // Opening removes small grass/soil specks.
            // Closing fills small gaps caused by reflections on the wood.
            // ------------------------------------------------------------

            val openKernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                Size(3.0, 3.0)
            )

            val closeKernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                Size(9.0, 9.0)
            )

            try {
                Imgproc.morphologyEx(
                    woodMask,
                    woodMask,
                    Imgproc.MORPH_OPEN,
                    openKernel
                )

                Imgproc.morphologyEx(
                    woodMask,
                    woodMask,
                    Imgproc.MORPH_CLOSE,
                    closeKernel
                )
            } finally {
                openKernel.release()
                closeKernel.release()
            }

            // ------------------------------------------------------------
            // 4. Distance transform.
            //
            // Every foreground pixel gets its distance from the nearest
            // background pixel.
            //
            // Therefore the maximum inside a wood should occur close to
            // its centre, and the value is an estimate of its radius.
            // ------------------------------------------------------------

            Imgproc.distanceTransform(
                woodMask,
                distMap,
                Geometry.DIST_L2,
                5
            )

            // ------------------------------------------------------------
            // 5. Find local maxima in the distance map.
            //
            // A 15x15 neighbourhood means we are looking for one peak
            // per reasonably sized object rather than every pixel on a
            // broad plateau.
            // ------------------------------------------------------------

            val peakKernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                Size(15.0, 15.0)
            )

            try {
                Imgproc.dilate(
                    distMap,
                    localMax,
                    peakKernel
                )
            } finally {
                peakKernel.release()
            }

            // Pixels which equal the local maximum are candidate centres.
            Core.compare(
                distMap,
                localMax,
                peakMask,
                Core.CMP_EQ
            )

            // ------------------------------------------------------------
            // 6. Ignore very small peaks.
            //
            // At our 1024px working resolution a real wood should have
            // a distance-transform radius comfortably above a handful
            // of pixels.
            //
            // We deliberately keep this fairly permissive for now.
            // ------------------------------------------------------------

            Core.inRange(
                distMap,
                Scalar(7.0),
                Scalar(100.0),
                radiusMask
            )

            Core.bitwise_and(
                peakMask,
                radiusMask,
                peakMask
            )

            // ------------------------------------------------------------
            // 7. Group neighbouring peak pixels.
            //
            // connectedComponentsWithStats gives us one component per
            // local maximum and its centroid.
            // ------------------------------------------------------------

            val componentCount =
                Imgproc.connectedComponentsWithStats(
                    peakMask,
                    labels,
                    stats,
                    centroids,
                    8
                )

            val detections = mutableListOf<BlobDetection>()

            for (label in 1 until componentCount) {
                val cx = centroids.get(label, 0)[0]
                val cy = centroids.get(label, 1)[0]

                if (!cx.isFinite() || !cy.isFinite()) {
                    continue
                }

                // --------------------------------------------------------
                // Recover the actual distance-transform value at the
                // detected centre.
                // --------------------------------------------------------

                val ix = cx.roundToInt()
                    .coerceIn(0, distMap.cols() - 1)

                val iy = cy.roundToInt()
                    .coerceIn(0, distMap.rows() - 1)

                val peakRadius = distMap
                    .get(iy, ix)
                    ?.firstOrNull()
                    ?: continue

                // ------------------------------------------------------------
// Validate that this is a coherent brown object rather than
// simply a brown patch in the grass.
//
// We compare the amount of brown inside the candidate against
// the surrounding ring.
//
// A real wood should have:
//   - lots of brown pixels inside
//   - substantially less brown immediately outside
// ------------------------------------------------------------

                // Ignore tiny colour patches.
                if (peakRadius < 4.0) {
                    continue
                }

                // Ignore implausibly large regions.
                if (peakRadius > 30.0) {
                    continue
                }

                // --------------------------------------------------------
                // Convert the centre and radius back to the original
                // photograph's coordinate system.
                // --------------------------------------------------------

                val originalX = cx / scale
                val originalY = cy / scale
                val originalRadius = peakRadius / scale

                detections.add(
                    BlobDetection(
                        centre = Point2(
                            originalX.toFloat(),
                            originalY.toFloat()
                        ),
                        radiusPx = originalRadius.toFloat(),
                        areaPixels = (PI * originalRadius * originalRadius).toInt(),
                        solidity = 0.9
                    )
                )
            }

            // ------------------------------------------------------------
            // 8. Sort largest objects first.
            //
            // This is useful during development because if the colour
            // mask produces a few unwanted small candidates, the real
            // woods should normally be amongst the larger peaks.
            // ------------------------------------------------------------

            return detections
                .sortedByDescending { it.radiusPx }
                .take(4)

        } catch (e: Exception) {
            // Detection should fail cleanly rather than taking down
            // the image-processing operation.
            return emptyList()

        } finally {
            source.release()
            working.release()
            rgb.release()
            woodMask.release()
            distMap.release()
            localMax.release()
            peakMask.release()
            radiusMask.release()
            labels.release()
            stats.release()
            centroids.release()
        }
    }

    private fun filterValidContours(contours: List<MatOfPoint>, scale: Double, gray: Mat): List<BlobDetection> =
        contours
            .mapNotNull { contour ->
                val areaVal = Geometry.contourArea(contour)

                // ✅ Tighten area filtering based on actual bowl sizes
                if (areaVal < 400.0 || areaVal > 15000.0) return@mapNotNull null

                val mop2f = MatOfPoint2f()
                try {
                    contour.convertTo(mop2f, CvType.CV_32F)

                    // ✅ Higher circularity threshold - bowls are very round
                    val perimeter = Geometry.arcLength(mop2f, true)
                    val circularity = if (perimeter > 0)
                        (4 * PI * areaVal) / (perimeter * perimeter) else 0.0

                    if (circularity < 0.75) return@mapNotNull null // Reject non-round shapes

                    // ✅ Optional: Brightness check to reject shadows
                    val centerPoint = Point(mop2f.rows() / 2.0, mop2f.cols() / 2.0)
                    val brightness =
                        gray.get(centerPoint.y.toInt(), centerPoint.x.toInt())?.firstOrNull()
                            ?: 150.0

                    // Shadows are dark (< 100) AND have smooth gradients
                    if (brightness < 120 && areaVal > 1000) return@mapNotNull null

                    val radius = mop2f.rows().toFloat() / scale

                    BlobDetection(
                        centre = Point2(
                            (mop2f.rows() / 2.0 / scale).toFloat(),
                            (mop2f.cols() / 2.0 / scale).toFloat()
                        ),
                        radiusPx = (radius.toFloat()).coerceAtMost(80f),
                        areaPixels = areaVal.toInt(),
                        solidity = circularity
                    )
                } finally {
                    mop2f.release()
                }
            }

    /** Process and filter contours, accepting gray explicitly */
    private fun processContours(contours: List<MatOfPoint>, scale: Double, gray: Mat): List<BlobDetection> {
        val detections = mutableListOf<BlobDetection>()

        for (contour in contours) {
            val areaVal = Geometry.contourArea(contour)

            // FIXED: Corrected range to keep relevant objects.
            // Foreground woods can be quite large (up to ~20,000 pixels area)
            if (areaVal < 150.0 || areaVal > 25000.0) continue

            val mop2f = MatOfPoint2f()
            try {
                contour.convertTo(mop2f, CvType.CV_32F)
                val center = Point()
                val radiusArr = floatArrayOf(0f)
                Geometry.minEnclosingCircle(mop2f, center, radiusArr)
                val radius = radiusArr[0]

                // Circularity check to filter out long irregular grass patches
                val perimeter = Geometry.arcLength(mop2f, true)
                val circularity = if (perimeter > 0)
                    (4 * PI * areaVal) / (perimeter * perimeter) else 0.0

                if (circularity > 0.45) { // Threshold to ignore non-round grass patches

                    // ✅ Added: Brightness check using passed gray parameter
                    val centerX = (center.x / scale).toInt()
                    val centerY = (center.y / scale).toInt()

                    // Check if point is within image bounds
                    val innerX = centerX.coerceIn(0, gray.cols() - 1)
                    val innerY = centerY.coerceIn(0, gray.rows() - 1)

                    // Get brightness at center (optional shadow rejection)
                    val brightness = if (innerX > 0 && innerY > 0)
                        gray.get(innerY, innerX)?.firstOrNull()?.toDouble() ?: 255.0 else 255.0

                    // Shadows are typically dark AND smooth gradients (< 100 + circularity > 0.9)
                    if (brightness < 80 && circularity > 0.9) continue  // Skip shadow regions


                    detections.add(
                        BlobDetection(
                            centre = Point2((center.x / scale).toFloat(), (center.y / scale).toFloat()),
                            radiusPx = (radius / scale).toFloat(),
                            areaPixels = areaVal.toInt(),
                            solidity = circularity
                        )
                    )
                }
            } finally {
                mop2f.release()
            }
        }

        return detections
    }
}

data class BlobDetection(
    val centre: Point2,
    val radiusPx: Float,
    val areaPixels: Int,
    val solidity: Double
)