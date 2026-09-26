package com.example.bowlsmeasuring

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import java.util.Locale
import kotlin.math.*

/**
 * Wood detector based on:
 *
 *   1. Heavy blur to suppress grass texture.
 *   2. Large-scale luminance structure to find candidate centres.
 *   3. Original image analysis around each candidate.
 *   4. Radial boundary detection.
 *   5. Ellipse fitting and boundary consistency.
 *
 * This detector deliberately does not use colour information, so it
 * can detect both black and brown woods.
 *
 * Coordinate convention:
 *
 *   detect()
 *       returns original bitmap coordinates.
 *
 *   detectNear()
 *       returns original bitmap coordinates.
 *
 * detectInRegion() NEVER changes the coordinate system.
 */
object WoodDetector {

    private const val TARGET_MAX_DIM = 1024.0

    private val BLUR_SIGMAS =
        doubleArrayOf(
            8.0,
            12.0,
            16.0,
            20.0
        )

    private val BLUR_KERNELS =
        intArrayOf(
            31,
            41,
            61,
            81
        )

    private const val MIN_CANDIDATE_RESPONSE = 7.0

    private const val MAX_CENTRES = 30

    private const val MIN_DIAMETER = 24.0
    private const val MAX_DIAMETER = 180.0

    private const val RADIAL_SAMPLES = 72

    private const val TWO_PI = 2.0 * PI

    private const val MIN_EDGE_RADIUS_FRACTION = 0.45
    private const val MAX_EDGE_RADIUS_FRACTION = 1.60

    private const val MIN_EDGE_STRENGTH = 7.0

    private const val MIN_BOUNDARY_POINTS = 28

    private const val MAX_DETECTIONS = 4

    private const val NEAR_ROI_SIZE = 360

    private const val MIN_FINAL_SCORE = 0.28

    private const val DUPLICATE_DISTANCE_FACTOR = 0.65

    /**
     * Result returned by detectNearWithDebug().
     *
     * debugImage is in ORIGINAL bitmap coordinates.
     */
    data class DebugResult(
        val detection: BlobDetection?,
        val lines: List<String>,
        val debugImage: Bitmap?
    )

    /**
     * Information retained for each radial boundary sample.
     */
    private data class RadialDebugPoint(
        val angleDegrees: Double,
        val radius: Double,
        val gradient: Double,
        val point: Point
    )

    /**
     * Fully automatic detection.
     *
     * Returns up to four wood candidates in original bitmap pixels.
     */
    fun detect(
        bitmap: Bitmap
    ): List<BlobDetection> {

        val source = Mat()

        Utils.bitmapToMat(
            bitmap,
            source
        )

        if (source.empty()) {
            source.release()
            return emptyList()
        }

        try {

            val scale =
                calculateScale(source)

            val working =
                resizeForDetection(
                    source,
                    scale
                )

            try {

                val centres =
                    findCandidateCentres(
                        working
                    )

                val candidates =
                    centres.mapNotNull { centre ->

                        refineCandidate(
                            original = source,
                            candidateX =
                                centre.x / scale,
                            candidateY =
                                centre.y / scale,
                            workingScale = scale
                        )
                    }

                return selectCandidates(
                    candidates
                )

            } finally {
                working.release()
            }

        } catch (_: Exception) {

            return emptyList()

        } finally {
            source.release()
        }
    }

    /**
     * Detect the wood nearest to a user click.
     *
     * All input/output coordinates are original bitmap coordinates.
     */
    fun detectNear(
        bitmap: Bitmap,
        clickedPoint: Point2
    ): BlobDetection? {

        val source = Mat()

        Utils.bitmapToMat(
            bitmap,
            source
        )

        if (source.empty()) {
            source.release()
            return null
        }

        try {

            val scale =
                calculateScale(source)

            val working =
                resizeForDetection(
                    source,
                    scale
                )

            try {

                val clickX =
                    clickedPoint.x * scale

                val clickY =
                    clickedPoint.y * scale

                val half =
                    NEAR_ROI_SIZE / 2

                val sx =
                    maxOf(
                        0,
                        (
                                clickX - half
                                ).roundToInt()
                    )

                val sy =
                    maxOf(
                        0,
                        (
                                clickY - half
                                ).roundToInt()
                    )

                val ex =
                    minOf(
                        working.cols(),
                        sx + NEAR_ROI_SIZE
                    )

                val ey =
                    minOf(
                        working.rows(),
                        sy + NEAR_ROI_SIZE
                    )

                val width =
                    ex - sx

                val height =
                    ey - sy

                if (
                    width < 40 ||
                    height < 40
                ) {
                    return null
                }

                val roi =
                    working.submat(
                        Rect(
                            sx,
                            sy,
                            width,
                            height
                        )
                    )

                try {

                    val centres =
                        findCandidateCentres(
                            roi
                        )

                    val candidates =
                        centres.mapNotNull { centre ->

                            val workingX =
                                centre.x + sx

                            val workingY =
                                centre.y + sy

                            refineCandidate(
                                original = source,
                                candidateX =
                                    workingX / scale,
                                candidateY =
                                    workingY / scale,
                                workingScale =
                                    scale
                            )
                        }

                    if (
                        candidates.isEmpty()
                    ) {
                        return null
                    }

                    return candidates
                        .minByOrNull { candidate ->

                            val dx =
                                candidate.detection
                                    .centre
                                    .x -
                                        clickedPoint.x

                            val dy =
                                candidate.detection
                                    .centre
                                    .y -
                                        clickedPoint.y

                            val distance =
                                sqrt(
                                    dx * dx +
                                            dy * dy
                                )

                            distance -
                                    candidate.score * 20.0
                        }
                        ?.detection

                } finally {
                    roi.release()
                }

            } finally {
                working.release()
            }

        } catch (_: Exception) {

            return null

        } finally {
            source.release()
        }
    }

    /**
     * Debug version of detectNear().
     *
     * The detection algorithm is deliberately the same as detectNear().
     *
     * In addition, this returns:
     *
     *   - detailed radial measurements
     *   - candidate measurements
     *   - a debug image showing the selected radial points
     *
     * The debug image uses ORIGINAL bitmap coordinates.
     */
    fun detectNearWithDebug(
        bitmap: Bitmap,
        clickedPoint: Point2
    ): DebugResult {

        val lines =
            mutableListOf<String>()

        lines +=
            "WoodDetector (Near click)"
        lines +=
            "----------------------------"
        lines +=
            "Image: ${bitmap.width} x ${bitmap.height}"
        lines +=
            String.format(
                Locale.US,
                "Click: (%.1f, %.1f)",
                clickedPoint.x,
                clickedPoint.y
            )

        val source = Mat()

        Utils.bitmapToMat(
            bitmap,
            source
        )

        if (source.empty()) {

            lines += "Source image is empty."

            return DebugResult(
                detection = null,
                lines = lines,
                debugImage = null
            )
        }

        /*
         * Work on a colour copy so that the debug annotations can be
         * drawn over the original photograph.
         */
        val debugMat =
            source.clone()

        try {

            val scale =
                calculateScale(source)

            val working =
                resizeForDetection(
                    source,
                    scale
                )

            try {

                lines +=
                    "Working image: " +
                            "${working.cols()} x ${working.rows()}"

                lines += ""

                /*
                 * Generate the same ROI as detectNear().
                 */
                val clickX =
                    clickedPoint.x * scale

                val clickY =
                    clickedPoint.y * scale

                val half =
                    NEAR_ROI_SIZE / 2

                val sx =
                    maxOf(
                        0,
                        (
                                clickX - half
                                ).roundToInt()
                    )

                val sy =
                    maxOf(
                        0,
                        (
                                clickY - half
                                ).roundToInt()
                    )

                val ex =
                    minOf(
                        working.cols(),
                        sx + NEAR_ROI_SIZE
                    )

                val ey =
                    minOf(
                        working.rows(),
                        sy + NEAR_ROI_SIZE
                    )

                val width =
                    ex - sx

                val height =
                    ey - sy

                if (
                    width < 40 ||
                    height < 40
                ) {

                    lines +=
                        "ROI too small."

                    return DebugResult(
                        detection = null,
                        lines = lines,
                        debugImage =
                            matToBitmap(debugMat)
                    )
                }

                val roi =
                    working.submat(
                        Rect(
                            sx,
                            sy,
                            width,
                            height
                        )
                    )

                try {

                    lines +=
                        "ROI: x=$sx y=$sy " +
                                "width=$width height=$height"

                    lines += ""

                    /*
                     * Candidate generation diagnostics.
                     */
                    val candidateDebug =
                        CandidateDebug()

                    val centres =
                        findCandidateCentres(
                            roi,
                            candidateDebug
                        )

                    lines +=
                        "Candidate regions:"
                    lines +=
                        "  contours = " +
                                candidateDebug.contours
                    lines +=
                        "  after area filter = " +
                                candidateDebug.afterAreaFilter
                    lines +=
                        "  candidate centres = " +
                                centres.size

                    lines += ""

                    if (
                        centres.isEmpty()
                    ) {

                        lines +=
                            "No candidate centres."

                        drawClickedPoint(
                            debugMat,
                            clickedPoint
                        )

                        return DebugResult(
                            detection = null,
                            lines = lines,
                            debugImage =
                                matToBitmap(debugMat)
                        )
                    }

                    val candidates =
                        mutableListOf<RefinedCandidate>()

                    centres.forEachIndexed { index, centre ->

                        val workingX =
                            centre.x + sx

                        val workingY =
                            centre.y + sy

                        val candidateX =
                            workingX / scale

                        val candidateY =
                            workingY / scale

                        val candidateResult =
                            refineCandidate(
                                original = source,
                                candidateX =
                                    candidateX,
                                candidateY =
                                    candidateY,
                                workingScale =
                                    scale,
                                debugLines =
                                    lines,
                                candidateIndex =
                                    index,
                                debugMat =
                                    debugMat
                            )

                        if (
                            candidateResult != null
                        ) {
                            candidates +=
                                candidateResult
                        }
                    }

                    lines += ""
                    lines +=
                        "Refined candidates = " +
                                candidates.size

                    if (
                        candidates.isEmpty()
                    ) {

                        lines +=
                            "Final woods = 0"

                        drawClickedPoint(
                            debugMat,
                            clickedPoint
                        )

                        return DebugResult(
                            detection = null,
                            lines = lines,
                            debugImage =
                                matToBitmap(debugMat)
                        )
                    }

                    val selected =
                        candidates
                            .minByOrNull { candidate ->

                                val dx =
                                    candidate.detection
                                        .centre
                                        .x -
                                            clickedPoint.x

                                val dy =
                                    candidate.detection
                                        .centre
                                        .y -
                                            clickedPoint.y

                                val distance =
                                    sqrt(
                                        dx * dx +
                                                dy * dy
                                    )

                                distance -
                                        candidate.score * 20.0
                            }

                    drawClickedPoint(
                        debugMat,
                        clickedPoint
                    )

                    if (
                        selected != null
                    ) {

                        drawSelectedCandidate(
                            debugMat,
                            selected
                        )

                        lines +=
                            "Final woods = 1"
                    } else {

                        lines +=
                            "Final woods = 0"
                    }

                    return DebugResult(
                        detection =
                            selected?.detection,
                        lines = lines,
                        debugImage =
                            matToBitmap(debugMat)
                    )

                } finally {
                    roi.release()
                }

            } finally {
                working.release()
            }

        } catch (e: Exception) {

            lines += ""
            lines +=
                "EXCEPTION: " +
                        (e.message ?: e.javaClass.simpleName)

            return DebugResult(
                detection = null,
                lines = lines,
                debugImage =
                    matToBitmap(debugMat)
            )

        } finally {

            debugMat.release()
            source.release()
        }
    }

    /**
     * Candidate generation.
     */
    private fun findCandidateCentres(
        source: Mat,
        debug: CandidateDebug? = null
    ): List<Point> {

        val gray = Mat()

        val response = Mat()

        val zeros =
            Mat.zeros(
                source.rows(),
                source.cols(),
                CvType.CV_8U
            )

        try {

            convertToGray(
                source,
                gray
            )

            Imgproc.GaussianBlur(
                gray,
                gray,
                Size(
                    5.0,
                    5.0
                ),
                1.5
            )

            for (
            index in BLUR_SIGMAS.indices
            ) {

                val blurred = Mat()
                val localBackground = Mat()
                val localResponse = Mat()

                try {

                    val kernel =
                        BLUR_KERNELS[index]

                    Imgproc.GaussianBlur(
                        gray,
                        blurred,
                        Size(
                            kernel.toDouble(),
                            kernel.toDouble()
                        ),
                        BLUR_SIGMAS[index]
                    )

                    Imgproc.GaussianBlur(
                        blurred,
                        localBackground,
                        Size(
                            (
                                    kernel * 1.7
                                    ).toInt()
                                .coerceAtLeast(3)
                                .let {
                                    if (it % 2 == 0) {
                                        (it + 1).toDouble()
                                    } else {
                                        it.toDouble()
                                    }
                                },
                            (
                                    kernel * 1.7
                                    ).toInt()
                                .coerceAtLeast(3)
                                .let {
                                    if (it % 2 == 0) {
                                        (it + 1).toDouble()
                                    } else {
                                        it.toDouble()
                                    }
                                }
                        ),
                        BLUR_SIGMAS[index] * 1.7
                    )

                    Core.absdiff(
                        blurred,
                        localBackground,
                        localResponse
                    )

                    Core.max(
                        zeros,
                        localResponse,
                        response
                    )

                } finally {

                    blurred.release()
                    localBackground.release()
                    localResponse.release()
                }
            }

            val responseMinMax =
                Core.minMaxLoc(response)

            if (debug != null) {
                debug.responseMin =
                    responseMinMax.minVal
                debug.responseMax =
                    responseMinMax.maxVal
                debug.responseMean =
                    Core.mean(response).`val`[0]
            }

            val peakKernel =
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(
                        25.0,
                        25.0
                    )
                )

            val localMax = Mat()
            val peakMask = Mat()

            try {

                Imgproc.dilate(
                    response,
                    localMax,
                    peakKernel
                )

                Core.compare(
                    response,
                    localMax,
                    peakMask,
                    Core.CMP_EQ
                )

                val thresholdMask =
                    Mat()

                try {

                    Imgproc.threshold(
                        response,
                        thresholdMask,
                        MIN_CANDIDATE_RESPONSE,
                        255.0,
                        Imgproc.THRESH_BINARY
                    )

                    thresholdMask.convertTo(
                        thresholdMask,
                        CvType.CV_8U
                    )

                    Core.bitwise_and(
                        peakMask,
                        thresholdMask,
                        peakMask
                    )

                } finally {
                    thresholdMask.release()
                }

                val contours =
                    ArrayList<MatOfPoint>()

                val hierarchy =
                    Mat()

                try {

                    Imgproc.findContours(
                        peakMask,
                        contours,
                        hierarchy,
                        Imgproc.RETR_EXTERNAL,
                        Imgproc.CHAIN_APPROX_SIMPLE
                    )

                    if (debug != null) {
                        debug.contours =
                            contours.size

                        /*
                         * This is the number that the existing
                         * implementation effectively retains before
                         * returning candidate centres.
                         */
                        debug.afterAreaFilter =
                            contours.size
                    }

                    return contours
                        .mapNotNull { contour ->

                            val moments =
                                Geometry.moments(
                                    contour
                                )

                            if (
                                abs(moments.m00) <
                                1e-9
                            ) {
                                null
                            } else {

                                Point(
                                    moments.m10 /
                                            moments.m00,

                                    moments.m01 /
                                            moments.m00
                                )
                            }
                        }
                        .sortedByDescending { point ->

                            response
                                .get(
                                    point.y
                                        .roundToInt()
                                        .coerceIn(
                                            0,
                                            response.rows() - 1
                                        ),

                                    point.x
                                        .roundToInt()
                                        .coerceIn(
                                            0,
                                            response.cols() - 1
                                        )
                                )
                                ?.firstOrNull()
                                ?: 0.0
                        }
                        .take(MAX_CENTRES)

                } finally {

                    hierarchy.release()

                    contours.forEach {
                        it.release()
                    }
                }

            } finally {

                peakKernel.release()
                localMax.release()
            }

        } finally {

            gray.release()
            response.release()
        }
    }

    /**
     * Refine one candidate using the ORIGINAL image.
     *
     * Optional debug arguments do not alter the detection itself.
     */
    private fun refineCandidate(
        original: Mat,
        candidateX: Double,
        candidateY: Double,
        workingScale: Double,
        debugLines: MutableList<String>? = null,
        candidateIndex: Int = -1,
        debugMat: Mat? = null
    ): RefinedCandidate? {

        if (
            candidateX < 0.0 ||
            candidateY < 0.0 ||
            candidateX >= original.cols() ||
            candidateY >= original.rows()
        ) {
            return null
        }

        val gray = Mat()

        try {

            convertToGray(
                original,
                gray
            )

            val expectedRadius =
                estimateInitialRadius(
                    original,
                    candidateX,
                    candidateY,
                    workingScale
                )

            val searchRadius =
                (
                        expectedRadius * 1.8
                        )
                    .roundToInt()
                    .coerceAtLeast(30)

            val sx =
                (
                        candidateX - searchRadius
                        )
                    .roundToInt()
                    .coerceAtLeast(0)

            val sy =
                (
                        candidateY - searchRadius
                        )
                    .roundToInt()
                    .coerceAtLeast(0)

            val ex =
                (
                        candidateX + searchRadius
                        )
                    .roundToInt()
                    .coerceAtMost(
                        gray.cols() - 1
                    )

            val ey =
                (
                        candidateY + searchRadius
                        )
                    .roundToInt()
                    .coerceAtMost(
                        gray.rows() - 1
                    )

            val width =
                ex - sx + 1

            val height =
                ey - sy + 1

            if (
                width < 30 ||
                height < 30
            ) {
                return null
            }

            val roi =
                gray.submat(
                    Rect(
                        sx,
                        sy,
                        width,
                        height
                    )
                )

            try {

                val localCentre =
                    Point(
                        candidateX - sx,
                        candidateY - sy
                    )

                val smoothed =
                    Mat()

                try {

                    Imgproc.GaussianBlur(
                        roi,
                        smoothed,
                        Size(
                            5.0,
                            5.0
                        ),
                        1.2
                    )

                    val boundary =
                        findRadialBoundary(
                            image = smoothed,
                            centre = localCentre,
                            expectedRadius =
                                expectedRadius,
                            debugLines =
                                debugLines,
                            candidateIndex =
                                candidateIndex
                        )
                            ?: return null

                    /*
                     * Draw the actual selected radial points on the
                     * ORIGINAL image.
                     */
                    if (
                        debugMat != null &&
                        boundary.radialDebugPoints != null
                    ) {

                        drawRadialDebugPoints(
                            debugMat,
                            boundary.radialDebugPoints,
                            sx,
                            sy
                        )
                    }

                    val ellipse =
                        fitBoundaryEllipse(
                            boundary.points
                        )
                            ?: return null

                    val major =
                        maxOf(
                            ellipse.size.width,
                            ellipse.size.height
                        )

                    val minor =
                        minOf(
                            ellipse.size.width,
                            ellipse.size.height
                        )

                    if (
                        major <= 0.0 ||
                        minor <= 0.0
                    ) {
                        return null
                    }

                    val aspect =
                        minor / major

                    val radius =
                        sqrt(
                            major * minor
                        ) / 2.0

                    if (
                        radius < 10.0 ||
                        radius > 250.0
                    ) {
                        return null
                    }

                    val fitScore =
                        ellipseFitScore(
                            boundary.points,
                            ellipse
                        )

                    val coverageScore =
                        boundary.coverage

                    val radialConsistency =
                        radialConsistencyScore(
                            boundary.points,
                            ellipse
                        )

                    val shapeScore =
                        (
                                0.40 *
                                        aspect.coerceIn(
                                            0.0,
                                            1.0
                                        ) +
                                        0.30 *
                                        fitScore +
                                        0.30 *
                                        radialConsistency
                                )
                            .coerceIn(
                                0.0,
                                1.0
                            )

                    if (
                        shapeScore <
                        MIN_FINAL_SCORE
                    ) {
                        return null
                    }

                    val edgeScore =
                        (
                                boundary.meanStrength /
                                        30.0
                                )
                            .coerceIn(
                                0.0,
                                1.0
                            )

                    val score =
                        0.50 * shapeScore +
                                0.25 * coverageScore +
                                0.15 * edgeScore +
                                0.10 * fitScore

                    val centreX =
                        ellipse.center.x + sx

                    val centreY =
                        ellipse.center.y + sy

                    val result =
                        RefinedCandidate(

                            detection =
                                BlobDetection(
                                    centre =
                                        Point2(
                                            centreX.toFloat(),
                                            centreY.toFloat()
                                        ),

                                    radiusPx =
                                        radius.toFloat(),

                                    areaPixels =
                                        (
                                                PI *
                                                        radius *
                                                        radius
                                                )
                                            .roundToInt(),

                                    solidity =
                                        shapeScore
                                ),

                            score = score,

                            radius = radius,

                            aspectRatio = aspect,

                            fitScore = fitScore,

                            coverage = coverageScore
                        )

                    if (
                        debugLines != null
                    ) {

                        debugLines +=
                            "Candidate $candidateIndex:"

                        debugLines +=
                            String.format(
                                Locale.US,
                                "  centre working = (%.1f, %.1f)",
                                candidateX *
                                        workingScale,
                                candidateY *
                                        workingScale
                            )

                        debugLines +=
                            String.format(
                                Locale.US,
                                "  centre original = (%.1f, %.1f)",
                                candidateX,
                                candidateY
                            )

                        debugLines +=
                            String.format(
                                Locale.US,
                                "  initial radius = %.1f",
                                expectedRadius
                            )

                        debugLines +=
                            "  radial points = " +
                                    boundary.points.size

                        debugLines +=
                            String.format(
                                Locale.US,
                                "  radius: min=%.1f max=%.1f mean=%.1f median=%.1f",
                                boundary.radialMin,
                                boundary.radialMax,
                                boundary.radialMean,
                                boundary.radialMedian
                            )

                        debugLines +=
                            String.format(
                                Locale.US,
                                "  ellipse coverage = %.2f",
                                coverageScore
                            )

                        debugLines +=
                            String.format(
                                Locale.US,
                                "  aspect ratio = %.2f",
                                aspect
                            )

                        debugLines +=
                            String.format(
                                Locale.US,
                                "  radial consistency = %.2f",
                                radialConsistency
                            )

                        debugLines +=
                            String.format(
                                Locale.US,
                                "  ellipse fit = %.2f",
                                fitScore
                            )

                        debugLines +=
                            String.format(
                                Locale.US,
                                "  edge strength = %.2f",
                                edgeScore
                            )

                        debugLines +=
                            String.format(
                                Locale.US,
                                "  final score = %.2f",
                                score
                            )

                        debugLines +=
                            "  FINAL = accepted"

                        debugLines +=
                            "  radial samples:"

                        boundary.radialDebugPoints
                            ?.forEach { point ->

                                debugLines +=
                                    String.format(
                                        Locale.US,
                                        "    %6.1f°  r=%6.1f  g=%8.2f  (%6.1f,%6.1f)",
                                        point.angleDegrees,
                                        point.radius,
                                        point.gradient,
                                        point.point.x + sx,
                                        point.point.y + sy
                                    )
                            }

                        debugLines += ""
                    }

                    /*
                     * Draw fitted ellipse as well.
                     */
                    if (
                        debugMat != null
                    ) {

                        Imgproc.ellipse(
                            debugMat,
                            RotatedRect(
                                Point(
                                    ellipse.center.x + sx,
                                    ellipse.center.y + sy
                                ),
                                ellipse.size,
                                ellipse.angle
                            ),
                            Scalar(
                                0.0,
                                255.0,
                                0.0
                            ),
                            2
                        )
                    }

                    return result

                } finally {
                    smoothed.release()
                }

            } finally {
                roi.release()
            }

        } finally {
            gray.release()
        }
    }

    /**
     * Estimate an initial radius.
     */
    private fun estimateInitialRadius(
        original: Mat,
        x: Double,
        y: Double,
        workingScale: Double
    ): Double {

        val imageScale =
            maxOf(
                original.cols(),
                original.rows()
            ).toDouble()

        val workingRadius =
            (
                    imageScale * workingScale
                    ) * 0.035

        return workingRadius
            .coerceIn(
                14.0,
                90.0
            ) / workingScale
    }

    /**
     * Search radially for the strongest boundary transition.
     *
     * Debug information records the selected point from each radial
     * direction.
     */
    private fun findRadialBoundary(
        image: Mat,
        centre: Point,
        expectedRadius: Double,
        debugLines: MutableList<String>? = null,
        candidateIndex: Int = -1
    ): BoundaryResult? {

        val points =
            mutableListOf<Point>()

        val radialDebug =
            mutableListOf<RadialDebugPoint>()

        var strengthSum =
            0.0

        var successful =
            0

        for (
        i in 0 until RADIAL_SAMPLES
        ) {

            val angle =
                TWO_PI *
                        i /
                        RADIAL_SAMPLES

            val dx =
                cos(angle)

            val dy =
                sin(angle)

            val minRadius =
                expectedRadius *
                        MIN_EDGE_RADIUS_FRACTION

            val maxRadius =
                expectedRadius *
                        MAX_EDGE_RADIUS_FRACTION

            var bestRadius =
                Double.NaN

            var bestStrength =
                0.0

            var radius =
                minRadius

            while (
                radius <= maxRadius
            ) {

                val x =
                    centre.x +
                            dx * radius

                val y =
                    centre.y +
                            dy * radius

                val previousX =
                    centre.x +
                            dx * (radius - 1.0)

                val previousY =
                    centre.y +
                            dy * (radius - 1.0)

                if (
                    !inside(
                        image,
                        x,
                        y
                    ) ||
                    !inside(
                        image,
                        previousX,
                        previousY
                    )
                ) {
                    break
                }

                val current =
                    sampleGray(
                        image,
                        x,
                        y
                    )

                val previous =
                    sampleGray(
                        image,
                        previousX,
                        previousY
                    )

                val gradient =
                    abs(
                        current - previous
                    )

                if (
                    gradient > bestStrength
                ) {

                    bestStrength =
                        gradient

                    bestRadius =
                        radius
                }

                radius += 1.0
            }

            if (
                bestRadius.isFinite() &&
                bestStrength >=
                MIN_EDGE_STRENGTH
            ) {

                val point =
                    Point(
                        centre.x +
                                dx * bestRadius,

                        centre.y +
                                dy * bestRadius
                    )

                points += point

                radialDebug +=
                    RadialDebugPoint(
                        angleDegrees =
                            Math.toDegrees(
                                angle
                            ),
                        radius =
                            bestRadius,
                        gradient =
                            bestStrength,
                        point =
                            point
                    )

                strengthSum +=
                    bestStrength

                successful++
            }
        }

        if (
            successful <
            MIN_BOUNDARY_POINTS
        ) {
            return null
        }

        val radii =
            radialDebug
                .map { it.radius }
                .sorted()

        val median =
            if (
                radii.size % 2 == 0
            ) {
                (
                        radii[
                            radii.size / 2 - 1
                        ] +
                                radii[
                                    radii.size / 2
                                ]
                        ) / 2.0
            } else {
                radii[
                    radii.size / 2
                ]
            }

        return BoundaryResult(

            points = points,

            coverage =
                successful.toDouble() /
                        RADIAL_SAMPLES,

            meanStrength =
                strengthSum /
                        successful,

            radialDebugPoints =
                radialDebug,

            radialMin =
                radii.minOrNull()
                    ?: 0.0,

            radialMax =
                radii.maxOrNull()
                    ?: 0.0,

            radialMean =
                radii.average(),

            radialMedian =
                median
        )
    }

    private fun fitBoundaryEllipse(
        points: List<Point>
    ): RotatedRect? {

        if (
            points.size < 5
        ) {
            return null
        }

        val contour =
            MatOfPoint2f(
                *points.toTypedArray()
            )

        try {

            return Geometry.fitEllipse(
                contour
            )

        } catch (_: Exception) {

            return null

        } finally {
            contour.release()
        }
    }

    private fun ellipseFitScore(
        points: List<Point>,
        ellipse: RotatedRect
    ): Double {

        val a =
            ellipse.size.width / 2.0

        val b =
            ellipse.size.height / 2.0

        if (
            a <= 0.0 ||
            b <= 0.0
        ) {
            return 0.0
        }

        val angle =
            Math.toRadians(
                ellipse.angle.toDouble()
            )

        val cosA =
            cos(angle)

        val sinA =
            sin(angle)

        var errorSum =
            0.0

        var count =
            0

        for (
        point in points
        ) {

            val dx =
                point.x -
                        ellipse.center.x

            val dy =
                point.y -
                        ellipse.center.y

            val ex =
                dx * cosA +
                        dy * sinA

            val ey =
                -dx * sinA +
                        dy * cosA

            val normalized =
                sqrt(
                    ex * ex /
                            (a * a) +
                            ey * ey /
                            (b * b)
                )

            val error =
                abs(
                    normalized - 1.0
                )

            errorSum += error
            count++
        }

        if (
            count == 0
        ) {
            return 0.0
        }

        val meanError =
            errorSum / count

        return (
                1.0 -
                        meanError / 0.35
                )
            .coerceIn(
                0.0,
                1.0
            )
    }

    private fun radialConsistencyScore(
        points: List<Point>,
        ellipse: RotatedRect
    ): Double {

        val a =
            ellipse.size.width / 2.0

        val b =
            ellipse.size.height / 2.0

        if (
            a <= 0.0 ||
            b <= 0.0
        ) {
            return 0.0
        }

        val angle =
            Math.toRadians(
                ellipse.angle.toDouble()
            )

        val cosA =
            cos(angle)

        val sinA =
            sin(angle)

        var errorSum =
            0.0

        var count =
            0

        for (
        point in points
        ) {

            val dx =
                point.x -
                        ellipse.center.x

            val dy =
                point.y -
                        ellipse.center.y

            val ex =
                dx * cosA +
                        dy * sinA

            val ey =
                -dx * sinA +
                        dy * cosA

            val radial =
                sqrt(
                    ex * ex +
                            ey * ey
                )

            val theta =
                atan2(
                    ey,
                    ex
                )

            val expected =
                a * b /
                        sqrt(
                            (
                                    b * cos(theta)
                                    ).pow(2) +
                                    (
                                            a * sin(theta)
                                            ).pow(2)
                        )

            if (
                expected > 0.0
            ) {

                errorSum +=
                    abs(
                        radial - expected
                    ) / expected

                count++
            }
        }

        if (
            count == 0
        ) {
            return 0.0
        }

        val meanError =
            errorSum / count

        return (
                1.0 -
                        meanError / 0.35
                )
            .coerceIn(
                0.0,
                1.0
            )
    }

    private fun selectCandidates(
        candidates: List<RefinedCandidate>
    ): List<BlobDetection> {

        val selected =
            mutableListOf<RefinedCandidate>()

        for (
        candidate in candidates.sortedByDescending {
            it.score
        }
        ) {

            val duplicate =
                selected.any { existing ->

                    val dx =
                        candidate.detection
                            .centre.x -
                                existing.detection
                                    .centre.x

                    val dy =
                        candidate.detection
                            .centre.y -
                                existing.detection
                                    .centre.y

                    val distance =
                        sqrt(
                            dx * dx +
                                    dy * dy
                        )

                    distance <
                            minOf(
                                candidate.radius,
                                existing.radius
                            ) *
                            DUPLICATE_DISTANCE_FACTOR
                }

            if (
                !duplicate
            ) {
                selected += candidate
            }

            if (
                selected.size >=
                MAX_DETECTIONS
            ) {
                break
            }
        }

        return selected
            .sortedByDescending {
                it.score
            }
            .map {
                it.detection
            }
    }

    private fun convertToGray(
        source: Mat,
        destination: Mat
    ) {

        when (
            source.channels()
        ) {

            4 -> Imgproc.cvtColor(
                source,
                destination,
                Imgproc.COLOR_RGBA2GRAY
            )

            3 -> Imgproc.cvtColor(
                source,
                destination,
                Imgproc.COLOR_RGB2GRAY
            )

            1 -> source.copyTo(
                destination
            )

            else ->
                throw IllegalArgumentException(
                    "Unsupported image channels: " +
                            source.channels()
                )
        }
    }

    private fun sampleGray(
        image: Mat,
        x: Double,
        y: Double
    ): Double {

        val ix =
            x.roundToInt()
                .coerceIn(
                    0,
                    image.cols() - 1
                )

        val iy =
            y.roundToInt()
                .coerceIn(
                    0,
                    image.rows() - 1
                )

        return image
            .get(
                iy,
                ix
            )
            ?.firstOrNull()
            ?: 0.0
    }

    private fun inside(
        image: Mat,
        x: Double,
        y: Double
    ): Boolean {

        return x >= 1.0 &&
                y >= 1.0 &&
                x < image.cols() - 1 &&
                y < image.rows() - 1
    }

    private fun resizeForDetection(
        source: Mat,
        scale: Double
    ): Mat {

        val result =
            Mat()

        if (
            scale == 1.0
        ) {

            source.copyTo(
                result
            )

        } else {

            Imgproc.resize(
                source,
                result,
                Size(
                    (
                            source.cols() * scale
                            )
                        .roundToInt()
                        .toDouble(),

                    (
                            source.rows() * scale
                            )
                        .roundToInt()
                        .toDouble()
                )
            )
        }

        return result
    }

    private fun calculateScale(
        source: Mat
    ): Double {

        val maximum =
            maxOf(
                source.cols(),
                source.rows()
            ).toDouble()

        return if (
            maximum > TARGET_MAX_DIM
        ) {

            TARGET_MAX_DIM / maximum

        } else {

            1.0
        }
    }

    /**
     * Draw all selected radial edge points.
     *
     * The points are supplied in ROI-local ORIGINAL-image coordinates,
     * so sx/sy convert them back to the full original image.
     */
    private fun drawRadialDebugPoints(
        image: Mat,
        points: List<RadialDebugPoint>,
        sx: Int,
        sy: Int
    ) {

        for (
        point in points
        ) {

            Imgproc.circle(
                image,
                Point(
                    point.point.x + sx,
                    point.point.y + sy
                ),
                3,
                Scalar(
                    255.0,
                    0.0,
                    0.0
                ),
                -1
            )
        }
    }

    private fun drawClickedPoint(
        image: Mat,
        point: Point2
    ) {

        Imgproc.circle(
            image,
            Point(
                point.x.toDouble(),
                point.y.toDouble()
            ),
            6,
            Scalar(
                0.0,
                255.0,
                255.0
            ),
            2
        )
    }

    private fun drawSelectedCandidate(
        image: Mat,
        candidate: RefinedCandidate
    ) {

        val centre =
            Point(
                candidate.detection
                    .centre.x.toDouble(),
                candidate.detection
                    .centre.y.toDouble()
            )

        Imgproc.circle(
            image,
            centre,
            6,
            Scalar(
                0.0,
                255.0,
                0.0
            ),
            2
        )
    }

    private fun matToBitmap(
        mat: Mat
    ): Bitmap {

        val bitmap =
            Bitmap.createBitmap(
                mat.cols(),
                mat.rows(),
                Bitmap.Config.ARGB_8888
            )

        Utils.matToBitmap(
            mat,
            bitmap
        )

        return bitmap
    }

    private data class CandidateDebug(
        var contours: Int = 0,
        var afterAreaFilter: Int = 0,
        var responseMin: Double = 0.0,
        var responseMax: Double = 0.0,
        var responseMean: Double = 0.0
    )

    private data class BoundaryResult(

        val points: List<Point>,

        val coverage: Double,

        val meanStrength: Double,

        val radialDebugPoints:
        List<RadialDebugPoint>? = null,

        val radialMin: Double = 0.0,

        val radialMax: Double = 0.0,

        val radialMean: Double = 0.0,

        val radialMedian: Double = 0.0
    )

    private data class RefinedCandidate(

        val detection: BlobDetection,

        val score: Double,

        val radius: Double,

        val aspectRatio: Double,

        val fitScore: Double,

        val coverage: Double
    )
}