package com.example.bowlsmeasuring

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
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

    /*
     * Maximum dimension used during candidate generation.
     *
     * The original image is retained separately for final boundary
     * refinement.
     */
    private const val TARGET_MAX_DIM = 1024.0

    /*
     * Heavy blur scales.
     *
     * These are deliberately much stronger than the small blur used
     * by the previous detector. Their purpose is to make grass texture
     * disappear while retaining objects the size of woods.
     */
    private val BLUR_SIGMAS =
        doubleArrayOf(
            8.0,
            12.0,
            16.0,
            20.0
        )

    /*
     * Gaussian kernel sizes corresponding approximately to the
     * sigma values above.
     */
    private val BLUR_KERNELS =
        intArrayOf(
            31,
            41,
            61,
            81
        )

    /*
     * Candidate response threshold.
     *
     * This is deliberately fairly low because candidate generation
     * is only the first stage.
     */
    private const val MIN_CANDIDATE_RESPONSE = 7.0

    /*
     * Number of candidate centres retained before detailed analysis.
     */
    private const val MAX_CENTRES = 30

    /*
     * Approximate minimum/maximum wood diameter in working pixels.
     *
     * These are deliberately broad because camera distance varies.
     */
    private const val MIN_DIAMETER = 24.0
    private const val MAX_DIAMETER = 180.0

    /*
     * Number of radial samples used to find the boundary.
     */
    private const val RADIAL_SAMPLES = 72

    /*
     * Angular sampling step in radians.
     */
    private const val TWO_PI = 2.0 * PI

    /*
     * Minimum and maximum fraction of the estimated radius at which
     * an edge is allowed to occur.
     */
    private const val MIN_EDGE_RADIUS_FRACTION = 0.45
    private const val MAX_EDGE_RADIUS_FRACTION = 1.60

    /*
     * Minimum boundary-gradient strength.
     */
    private const val MIN_EDGE_STRENGTH = 7.0

    /*
     * Minimum number of successful radial edge measurements required
     * to fit an ellipse.
     */
    private const val MIN_BOUNDARY_POINTS = 28

    /*
     * Maximum number of returned woods.
     */
    private const val MAX_DETECTIONS = 4

    /*
     * Local search size for detectNear().
     */
    private const val NEAR_ROI_SIZE = 360

    /*
     * Minimum score for a refined candidate.
     */
    private const val MIN_FINAL_SCORE = 0.28

    /*
     * Radius used to suppress duplicate detections.
     */
    private const val DUPLICATE_DISTANCE_FACTOR = 0.65

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

                /*
                 * Candidate generation operates on the heavily blurred
                 * working image.
                 */
                val centres =
                    findCandidateCentres(
                        working
                    )

                /*
                 * Refinement uses the original image.
                 *
                 * We pass the original image, together with the scale
                 * used for the candidate centres.
                 */
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
                    clickedPoint.x *
                            scale

                val clickY =
                    clickedPoint.y *
                            scale

                val half =
                    NEAR_ROI_SIZE / 2

                val sx =
                    maxOf(
                        0,
                        (
                                clickX -
                                        half
                                ).roundToInt()
                    )

                val sy =
                    maxOf(
                        0,
                        (
                                clickY -
                                        half
                                ).roundToInt()
                    )

                val ex =
                    minOf(
                        working.cols(),
                        sx +
                                NEAR_ROI_SIZE
                    )

                val ey =
                    minOf(
                        working.rows(),
                        sy +
                                NEAR_ROI_SIZE
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

                /*
                 * Candidate generation happens in the ROI.
                 */
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

                    val localClick =
                        Point(
                            clickX - sx,
                            clickY - sy
                        )

                    val centres =
                        findCandidateCentres(
                            roi
                        )

                    /*
                     * Convert ROI-local candidate centres to original
                     * image coordinates before refinement.
                     */
                    val candidates =
                        centres.mapNotNull { centre ->

                            val workingX =
                                centre.x +
                                        sx

                            val workingY =
                                centre.y +
                                        sy

                            refineCandidate(
                                original = source,

                                candidateX =
                                    workingX /
                                            scale,

                                candidateY =
                                    workingY /
                                            scale,

                                workingScale =
                                    scale
                            )
                        }

                    if (
                        candidates.isEmpty()
                    ) {
                        return null
                    }

                    /*
                     * In near mode proximity to the tap is deliberately
                     * strong, but the candidate still has to look like
                     * a wood.
                     */
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

                            /*
                             * Score is used as a secondary term.
                             */
                            distance -
                                    candidate.score *
                                    20.0
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
     * Candidate generation.
     *
     * This is intentionally NOT trying to find the edge of the wood.
     *
     * It only asks:
     *
     *   "Where are there large-scale image structures that could
     *    plausibly be a wood-sized object?"
     *
     * The original image is used later to determine the boundary.
     */
    private fun findCandidateCentres(
        source: Mat
    ): List<Point> {

        val gray =
            Mat()

        val response =
            Mat.zeros(
                source.rows(),
                source.cols(),
                CvType.CV_32F
            )

        try {

            convertToGray(
                source,
                gray
            )

            /*
             * A little initial smoothing removes sensor-level noise.
             */
            Imgproc.GaussianBlur(
                gray,
                gray,
                Size(
                    5.0,
                    5.0
                ),
                1.5
            )

            /*
             * Accumulate the strongest large-scale response.
             */
            for (
            index in BLUR_SIGMAS.indices
            ) {

                val blurred =
                    Mat()

                val localBackground =
                    Mat()

                val localResponse =
                    Mat()

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

                    /*
                     * Difference between a moderately smoothed image
                     * and a much more heavily smoothed image.
                     *
                     * This suppresses fine grass texture.
                     */
                    Imgproc.GaussianBlur(
                        blurred,
                        localBackground,
                        Size(
                            (
                                    kernel *
                                            1.7
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
                                    kernel *
                                            1.7
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
                        BLUR_SIGMAS[index] *
                                1.7
                    )

                    Core.absdiff(
                        blurred,
                        localBackground,
                        localResponse
                    )

                    /*
                     * Keep the strongest response seen at each pixel.
                     */
                    Core.max(
                        response,
                        localResponse,
                        response
                    )

                } finally {

                    blurred.release()
                    localBackground.release()
                    localResponse.release()
                }
            }

            /*
             * Suppress tiny local maxima.
             */
            val peakKernel =
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(
                        25.0,
                        25.0
                    )
                )

            val localMax =
                Mat()

            val peakMask =
                Mat()

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

                /*
                 * Threshold the response.
                 */
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

            } finally {

                peakKernel.release()
                localMax.release()
            }

            /*
             * Extract the strongest peak locations.
             */
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

            gray.release()
            response.release()
        }
    }

    /**
     * Refine one candidate using the ORIGINAL image.
     *
     * This is the important second stage.
     */
    private fun refineCandidate(
        original: Mat,
        candidateX: Double,
        candidateY: Double,
        workingScale: Double
    ): RefinedCandidate? {

        if (
            candidateX < 0.0 ||
            candidateY < 0.0 ||
            candidateX >= original.cols() ||
            candidateY >= original.rows()
        ) {
            return null
        }

        val gray =
            Mat()

        try {

            convertToGray(
                original,
                gray
            )

            /*
             * We don't need to process the whole original image.
             *
             * Work in a local window around the candidate.
             */
            val expectedRadius =
                estimateInitialRadius(
                    original,
                    candidateX,
                    candidateY,
                    workingScale
                )

            val searchRadius =
                (
                        expectedRadius *
                                1.8
                        )
                    .roundToInt()
                    .coerceAtLeast(30)

            val sx =
                (
                        candidateX -
                                searchRadius
                        )
                    .roundToInt()
                    .coerceAtLeast(0)

            val sy =
                (
                        candidateY -
                                searchRadius
                        )
                    .roundToInt()
                    .coerceAtLeast(0)

            val ex =
                (
                        candidateX +
                                searchRadius
                        )
                    .roundToInt()
                    .coerceAtMost(
                        gray.cols() - 1
                    )

            val ey =
                (
                        candidateY +
                                searchRadius
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

                /*
                 * Light smoothing ONLY for edge measurement.
                 *
                 * We deliberately do not apply the huge blur here.
                 */
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
                                expectedRadius
                        )
                            ?: return null

                    val ellipse =
                        fitBoundaryEllipse(
                            boundary.points
                        )
                            ?: return null

                    /*
                     * Convert ellipse dimensions back to original
                     * image coordinates. We are already in original
                     * image pixels here, so no scale is required.
                     */
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
                        minor /
                                major

                    /*
                     * The candidate's apparent radius is the geometric
                     * mean of the fitted ellipse axes.
                     */
                    val radius =
                        sqrt(
                            major *
                                    minor
                        ) / 2.0

                    if (
                        radius < 10.0 ||
                        radius > 250.0
                    ) {
                        return null
                    }

                    /*
                     * How consistently do the measured boundary points
                     * lie on the fitted ellipse?
                     */
                    val fitScore =
                        ellipseFitScore(
                            boundary.points,
                            ellipse
                        )

                    /*
                     * How complete is the detected boundary?
                     */
                    val coverageScore =
                        boundary.coverage

                    /*
                     * How consistent are the radial distances?
                     *
                     * A shadow generally produces a highly asymmetric
                     * distribution of edge distances.
                     */
                    val radialConsistency =
                        radialConsistencyScore(
                            boundary.points,
                            ellipse
                        )

                    /*
                     * Compactness/shape score.
                     *
                     * We allow substantial perspective distortion.
                     */
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

                    /*
                     * Boundary gradient strength.
                     */
                    val edgeScore =
                        (
                                boundary.meanStrength /
                                        30.0
                                )
                            .coerceIn(
                                0.0,
                                1.0
                            )

                    /*
                     * Final candidate score.
                     *
                     * Geometry is deliberately the dominant signal.
                     */
                    val score =
                        0.50 *
                                shapeScore +
                                0.25 *
                                coverageScore +
                                0.15 *
                                edgeScore +
                                0.10 *
                                fitScore

                    /*
                     * Convert local ellipse centre back to complete
                     * original-image coordinates.
                     */
                    val centreX =
                        ellipse.center.x +
                                sx

                    val centreY =
                        ellipse.center.y +
                                sy

                    return RefinedCandidate(

                        detection =
                            BlobDetection(
                                centre =
                                    Point2(
                                        centreX
                                            .toFloat(),

                                        centreY
                                            .toFloat()
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
     *
     * At this stage this is deliberately broad. Later we can replace
     * this with the jack-derived estimate.
     */
    private fun estimateInitialRadius(
        original: Mat,
        x: Double,
        y: Double,
        workingScale: Double
    ): Double {

        /*
         * Start from a sensible fraction of the working-image size.
         *
         * The radial search itself can expand/contract from this value.
         */
        val imageScale =
            maxOf(
                original.cols(),
                original.rows()
            ).toDouble()

        /*
         * Typical wood radius at a 1024px working image.
         */
        val workingRadius =
            (
                    imageScale *
                            workingScale
                    ) *
                    0.035

        return workingRadius
            .coerceIn(
                14.0,
                90.0
            ) /
                workingScale
    }

    /**
     * Search radially for the strongest boundary transition.
     *
     * This is where we deliberately return to the original image.
     */
    private fun findRadialBoundary(
        image: Mat,
        centre: Point,
        expectedRadius: Double
    ): BoundaryResult? {

        val points =
            mutableListOf<Point>()

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

            /*
             * Sample every 1 pixel along the radial line.
             */
            var radius =
                minRadius

            while (
                radius <= maxRadius
            ) {

                val x =
                    centre.x +
                            dx *
                            radius

                val y =
                    centre.y +
                            dy *
                            radius

                val previousX =
                    centre.x +
                            dx *
                            (radius - 1.0)

                val previousY =
                    centre.y +
                            dy *
                            (radius - 1.0)

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
                        current -
                                previous
                    )

                if (
                    gradient >
                    bestStrength
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

                points +=
                    Point(
                        centre.x +
                                dx *
                                bestRadius,

                        centre.y +
                                dy *
                                bestRadius
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

        return BoundaryResult(

            points = points,

            coverage =
                successful.toDouble() /
                        RADIAL_SAMPLES,

            meanStrength =
                strengthSum /
                        successful
        )
    }

    /**
     * Fit an ellipse to the radial boundary points.
     */
    private fun fitBoundaryEllipse(
        points: List<Point>
    ): RotatedRect? {

        if (
            points.size < 5
        ) {
            return null
        }

        val pointArray =
            points.toTypedArray()

        val contour =
            MatOfPoint2f(
                *pointArray
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

    /**
     * Measures how closely the boundary points lie on the fitted
     * ellipse.
     */
    private fun ellipseFitScore(
        points: List<Point>,
        ellipse: RotatedRect
    ): Double {

        val a =
            ellipse.size.width /
                    2.0

        val b =
            ellipse.size.height /
                    2.0

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

            /*
             * Rotate point into ellipse coordinates.
             */
            val ex =
                dx *
                        cosA +
                        dy *
                        sinA

            val ey =
                -dx *
                        sinA +
                        dy *
                        cosA

            val normalized =
                sqrt(
                    (
                            ex * ex /
                                    (a * a)
                            ) +
                            (
                                    ey * ey /
                                            (b * b)
                                    )
                )

            val error =
                abs(
                    normalized - 1.0
                )

            errorSum +=
                error

            count++
        }

        if (
            count == 0
        ) {
            return 0.0
        }

        val meanError =
            errorSum /
                    count

        return (
                1.0 -
                        meanError /
                        0.35
                )
            .coerceIn(
                0.0,
                1.0
            )
    }

    /**
     * Measures how consistently the boundary surrounds the candidate.
     *
     * We compare the radial distances of the detected points with the
     * fitted ellipse rather than simply measuring circularity.
     */
    private fun radialConsistencyScore(
        points: List<Point>,
        ellipse: RotatedRect
    ): Double {

        val a =
            ellipse.size.width /
                    2.0

        val b =
            ellipse.size.height /
                    2.0

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
                dx *
                        cosA +
                        dy *
                        sinA

            val ey =
                -dx *
                        sinA +
                        dy *
                        cosA

            val radial =
                sqrt(
                    ex * ex +
                            ey * ey
                )

            /*
             * Expected radius of the ellipse in this direction.
             */
            val theta =
                atan2(
                    ey,
                    ex
                )

            val expected =
                (
                        a * b
                        ) /
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
                        radial -
                                expected
                    ) /
                            expected

                count++
            }
        }

        if (
            count == 0
        ) {
            return 0.0
        }

        val meanError =
            errorSum /
                    count

        return (
                1.0 -
                        meanError /
                        0.35
                )
            .coerceIn(
                0.0,
                1.0
            )
    }

    /**
     * Remove duplicate candidates and retain the four strongest.
     */
    private fun selectCandidates(
        candidates: List<RefinedCandidate>
    ): List<BlobDetection> {

        val selected =
            mutableListOf<RefinedCandidate>()

        for (
        candidate
        in candidates.sortedByDescending {
            it.score
        }
        ) {

            val duplicate =
                selected.any { existing ->

                    val dx =
                        candidate.detection
                            .centre
                            .x -
                                existing.detection
                                    .centre
                                    .x

                    val dy =
                        candidate.detection
                            .centre
                            .y -
                                existing.detection
                                    .centre
                                    .y

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
                selected +=
                    candidate
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

    /**
     * Convert Mat to greyscale.
     */
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

    /**
     * Bilinear-ish local greyscale sample using nearest pixel.
     *
     * For the boundary search this is sufficient because the image
     * has already had a small 5px Gaussian blur applied.
     */
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
                x <
                image.cols() - 1 &&
                y <
                image.rows() - 1
    }

    /**
     * Resize the complete image once.
     */
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
                            source.cols() *
                                    scale
                            ).roundToInt()
                        .toDouble(),

                    (
                            source.rows() *
                                    scale
                            ).roundToInt()
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
            maximum >
            TARGET_MAX_DIM
        ) {

            TARGET_MAX_DIM /
                    maximum

        } else {

            1.0
        }
    }

    private data class BoundaryResult(

        val points: List<Point>,

        val coverage: Double,

        val meanStrength: Double
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