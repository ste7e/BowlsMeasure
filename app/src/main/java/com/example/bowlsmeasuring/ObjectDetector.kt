package com.example.bowlsmeasuring

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/**
 * Semi-automatic detector for a single physical object.
 *
 * The user supplies an approximate click position and this detector
 * attempts to identify the physical object nearest to that click.
 *
 * The detector deliberately does NOT:
 *
 *   - identify the object as a jack or wood
 *   - use colour
 *   - assume a particular jack colour
 *   - assume a particular apparent jack/wood size relationship
 *
 * Those decisions belong to later stages of the measurement process.
 *
 * Detection strategy:
 *
 *   1. Resize the image to a manageable working size.
 *   2. Apply several heavy blur scales.
 *   3. Find large-scale candidate centres.
 *   4. Retain the scale at which each candidate was found.
 *   5. Return to the original image.
 *   6. Search for a closed boundary around the candidate at that scale.
 *   7. Fit an ellipse to the boundary.
 *   8. Prefer candidates close to the user's click.
 *
 * Coordinates returned by this class are always original bitmap pixels.
 */
object ObjectDetector {

    /*
     * Maximum working-image dimension.
     */
    private const val TARGET_MAX_DIM = 1024.0

    /*
     * Multiple scales are important because apparent object size
     * varies with perspective and camera distance.
     */
    private val BLUR_SIGMAS =
        doubleArrayOf(
            6.0,
            8.0,
            11.0,
            15.0,
            20.0,
            26.0
        )

    private val BLUR_KERNELS =
        intArrayOf(
            25,
            31,
            41,
            51,
            71,
            91
        )

    /*
     * Gaussian background is deliberately broader than the object
     * blur. Their difference gives us large-scale structure while
     * suppressing fine grass texture.
     */
    private const val BACKGROUND_SCALE = 1.7

    /*
     * Candidate response threshold.
     *
     * This is deliberately low. Candidate selection is subsequently
     * constrained by the user's click and boundary validation.
     */
    private const val MIN_CANDIDATE_RESPONSE = 5.0

    /*
     * Number of candidate centres retained from each scale.
     */
    private const val MAX_CANDIDATES_PER_SCALE = 12

    /*
     * Total candidates retained before detailed refinement.
     */
    private const val MAX_CANDIDATES = 30

    /*
     * Estimated relationship between Gaussian scale and object radius.
     *
     * This is NOT a jack/wood ratio.
     *
     * It merely converts the image-processing scale which generated
     * a blob into an approximate apparent object radius.
     *
     * Refinement searches around this value, so this does not have
     * to be exact.
     */
    private const val SCALE_TO_RADIUS = 1.5

    /*
     * Radial samples used around each candidate.
     */
    private const val RADIAL_SAMPLES = 72

    private const val TWO_PI = 2.0 * PI

    /*
     * Boundary search range relative to the scale-derived radius.
     */
    private const val MIN_EDGE_RADIUS_FRACTION = 0.60
    private const val MAX_EDGE_RADIUS_FRACTION = 1.45

    /*
     * How strongly the radial search favours an edge near the
     * candidate's expected radius.
     */
    private const val EDGE_RADIUS_TOLERANCE = 0.30

    /*
     * Minimum gradient strength for an accepted radial edge.
     */
    private const val MIN_EDGE_STRENGTH = 5.0

    /*
     * Minimum number of radial measurements needed for an ellipse.
     */
    private const val MIN_BOUNDARY_POINTS = 32

    /*
     * The click should be somewhere on or reasonably close to the
     * object.
     *
     * This is deliberately generous because the user is selecting
     * an area rather than accurately marking the centre.
     */
    private const val MAX_CLICK_DISTANCE_FACTOR = 1.35

    /*
     * Minimum final geometric quality.
     */
    private const val MIN_SHAPE_SCORE = 0.30

    /*
     * Minimum final score.
     */
    private const val MIN_FINAL_SCORE = 0.30

    /*
     * Local search region around the user's click, in working pixels.
     */
    private const val NEAR_ROI_SIZE = 360

    /**
     * Detect the physical object nearest to the user's click.
     *
     * The returned coordinates are in ORIGINAL bitmap pixels.
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

                /*
                 * Keep the ROI inside the working image.
                 */
                val sx =
                    (
                            clickX - half
                            )
                        .roundToInt()
                        .coerceIn(
                            0,
                            maxOf(
                                0,
                                working.cols() - 1
                            )
                        )

                val sy =
                    (
                            clickY - half
                            )
                        .roundToInt()
                        .coerceIn(
                            0,
                            maxOf(
                                0,
                                working.rows() - 1
                            )
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
                    width < 60 ||
                    height < 60
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

                    val localClick =
                        Point(
                            clickX - sx,
                            clickY - sy
                        )

                    val candidates =
                        findCandidateCentres(
                            roi
                        )

                    val candidatesWithClick = candidates

                    /*
                     * Remove obviously distant candidates before
                     * expensive original-image processing.
                     */
                    val nearbyCandidates =
                        candidatesWithClick
                            .distinctBy { candidate ->

                                /*
                                 * Coarse spatial key only. Multiple
                                 * scales are deliberately retained.
                                 */
                                "${candidate.centre.x.roundToInt() / 4}," +
                                        "${candidate.centre.y.roundToInt() / 4}," +
                                        "${candidate.estimatedRadius.roundToInt() / 4}"
                            }
                            .filter { candidate ->

                                val dx =
                                    candidate.centre.x -
                                            localClick.x

                                val dy =
                                    candidate.centre.y -
                                            localClick.y

                                val distance =
                                    sqrt(
                                        dx * dx +
                                                dy * dy
                                    )

                                /*
                                 * Allow the candidate to be outside
                                 * the object because the click may be
                                 * deliberately approximate.
                                 */
                                distance <=
                                        candidate.estimatedRadius *
                                        2.0 +
                                        25.0
                            }
                            .sortedBy { candidate ->

                                val dx =
                                    candidate.centre.x -
                                            localClick.x

                                val dy =
                                    candidate.centre.y -
                                            localClick.y

                                dx * dx + dy * dy
                            }
                            .take(MAX_CANDIDATES)

                    val refined =
                        nearbyCandidates.mapNotNull { candidate ->

                            val workingX =
                                candidate.centre.x + sx

                            val workingY =
                                candidate.centre.y + sy

                            val refinedCandidate =
                                refineCandidate(
                                    original = source,

                                    candidateX =
                                        workingX /
                                                scale,

                                    candidateY =
                                        workingY /
                                                scale,

                                    estimatedRadius =
                                        candidate.estimatedRadius /
                                                scale,

                                    clickedPoint =
                                        clickedPoint
                                )

                            refinedCandidate
                        }

                    /*
                     * The user's click is the primary piece of
                     * information.
                     *
                     * We therefore select by proximity first, with
                     * object quality acting as a secondary factor.
                     */
                    return refined
                        .maxByOrNull { candidate ->

                            /*
                             * Object quality is the primary criterion.
                             *
                             * Click distance is only used to break ties / favour
                             * candidates close to the selected area.
                             */
                            val proximity =
                                1.0 -
                                        candidate.clickDistance /
                                        100.0

                            candidate.score +
                                    proximity.coerceIn(0.0, 1.0) * 0.10
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
     * Find large-scale candidate centres.
     *
     * The important difference from the previous detector is that
     * each candidate retains the scale at which it was detected.
     */
    private fun findCandidateCentres(
        source: Mat
    ): List<CandidateCentre> {

        val gray =
            Mat()

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

            val candidates =
                mutableListOf<CandidateCentre>()

            for (
            index in BLUR_SIGMAS.indices
            ) {

                val blurred =
                    Mat()

                val background =
                    Mat()

                val difference =
                    Mat()

                val differenceFloat =
                    Mat()

                try {

                    val sigma =
                        BLUR_SIGMAS[index]

                    val kernel =
                        BLUR_KERNELS[index]

                    Imgproc.GaussianBlur(
                        gray,
                        blurred,
                        Size(
                            kernel.toDouble(),
                            kernel.toDouble()
                        ),
                        sigma
                    )

                    val backgroundKernel =
                        makeOddKernel(
                            (
                                    kernel *
                                            BACKGROUND_SCALE
                                    ).roundToInt()
                        )

                    Imgproc.GaussianBlur(
                        blurred,
                        background,
                        Size(
                            backgroundKernel.toDouble(),
                            backgroundKernel.toDouble()
                        ),
                        sigma *
                                BACKGROUND_SCALE
                    )

                    /*
                     * blurred/background are 8-bit images.
                     */
                    Core.absdiff(
                        blurred,
                        background,
                        difference
                    )

                    /*
                     * Explicit conversion is important because our
                     * accumulated response is CV_32F.
                     */
                    difference.convertTo(
                        differenceFloat,
                        CvType.CV_32F
                    )

                    val peakKernel =
                        Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE,
                            Size(
                                17.0,
                                17.0
                            )
                        )

                    val localMax =
                        Mat()

                    val thresholdMask =
                        Mat()

                    val peakMask =
                        Mat()

                    try {

                        Imgproc.dilate(
                            differenceFloat,
                            localMax,
                            peakKernel
                        )

                        Core.compare(
                            differenceFloat,
                            localMax,
                            peakMask,
                            Core.CMP_EQ
                        )

                        Imgproc.threshold(
                            differenceFloat,
                            thresholdMask,
                            MIN_CANDIDATE_RESPONSE,
                            255.0,
                            Imgproc.THRESH_BINARY
                        )

                        /*
                         * threshold output is CV_32F here.
                         */
                        thresholdMask.convertTo(
                            thresholdMask,
                            CvType.CV_8U
                        )

                        Core.bitwise_and(
                            peakMask,
                            thresholdMask,
                            peakMask
                        )

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

                            val scaleRadius =
                                sigma *
                                        SCALE_TO_RADIUS

                            contours
                                .mapNotNull { contour ->

                                    val moments =
                                        Geometry.moments(
                                            contour
                                        )

                                    if (
                                        abs(
                                            moments.m00
                                        ) < 1e-9
                                    ) {
                                        null
                                    } else {

                                        val x =
                                            moments.m10 /
                                                    moments.m00

                                        val y =
                                            moments.m01 /
                                                    moments.m00

                                        val ix =
                                            x.roundToInt()
                                                .coerceIn(
                                                    0,
                                                    differenceFloat.cols() - 1
                                                )

                                        val iy =
                                            y.roundToInt()
                                                .coerceIn(
                                                    0,
                                                    differenceFloat.rows() - 1
                                                )

                                        val response =
                                            differenceFloat
                                                .get(
                                                    iy,
                                                    ix
                                                )
                                                ?.firstOrNull()
                                                ?: 0.0

                                        CandidateCentre(
                                            centre =
                                                Point(
                                                    x,
                                                    y
                                                ),

                                            estimatedRadius =
                                                scaleRadius,

                                            response =
                                                response
                                        )
                                    }
                                }
                                .sortedByDescending {
                                    it.response
                                }
                                .take(
                                    MAX_CANDIDATES_PER_SCALE
                                )
                                .forEach {
                                    candidates += it
                                }

                        } finally {

                            hierarchy.release()

                            contours.forEach {
                                it.release()
                            }
                        }

                    } finally {

                        peakKernel.release()
                        localMax.release()
                        thresholdMask.release()
                        peakMask.release()
                    }

                } finally {

                    blurred.release()
                    background.release()
                    difference.release()
                    differenceFloat.release()
                }
            }

            return candidates
                .sortedByDescending {
                    it.response
                }
                .take(
                    MAX_CANDIDATES
                )

        } finally {
            gray.release()
        }
    }

    /**
     * Refine one candidate using the ORIGINAL image.
     */
    private fun refineCandidate(
        original: Mat,
        candidateX: Double,
        candidateY: Double,
        estimatedRadius: Double,
        clickedPoint: Point2
    ): RefinedCandidate? {

        if (
            candidateX < 0.0 ||
            candidateY < 0.0 ||
            candidateX >= original.cols() ||
            candidateY >= original.rows()
        ) {
            return null
        }

        val dx =
            candidateX -
                    clickedPoint.x

        val dy =
            candidateY -
                    clickedPoint.y

        val clickDistance =
            sqrt(
                dx * dx +
                        dy * dy
            )

        /*
         * Reject candidates that are clearly unrelated to the
         * user's click.
         */
        if (
            clickDistance >
            estimatedRadius *
            MAX_CLICK_DISTANCE_FACTOR +
            30.0
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
             * Search window based on the candidate's apparent scale.
             *
             * Unlike the previous implementation this is not derived
             * from the entire image dimensions.
             */
            val searchRadius =
                (
                        estimatedRadius *
                                1.7
                        )
                    .roundToInt()
                    .coerceAtLeast(25)

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

                val smoothed =
                    Mat()

                try {

                    /*
                     * Only light smoothing here. We want the actual
                     * object boundary, not the large-scale candidate
                     * structure.
                     */
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
                                estimatedRadius
                        )
                            ?: return null

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

                    /*
                     * Reject extremely elongated structures.
                     *
                     * We allow quite a lot of perspective distortion.
                     */
                    if (
                        aspect < 0.45
                    ) {
                        return null
                    }

                    val radius =
                        sqrt(
                            major *
                                    minor
                        ) / 2.0

                    if (
                        radius < 5.0 ||
                        radius > 400.0
                    ) {
                        return null
                    }

                    val fitScore =
                        ellipseFitScore(
                            boundary.points,
                            ellipse
                        )

                    val radialConsistency =
                        radialConsistencyScore(
                            boundary.points,
                            ellipse
                        )

                    val shapeScore =
                        (
                                0.35 *
                                        aspect.coerceIn(
                                            0.0,
                                            1.0
                                        ) +
                                        0.35 *
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
                        MIN_SHAPE_SCORE
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

                    /*
                     * How close is the fitted object centre to the
                     * user's click?
                     */
                    val fittedCentreX =
                        ellipse.center.x + sx

                    val fittedCentreY =
                        ellipse.center.y + sy

                    val fittedDx =
                        fittedCentreX -
                                clickedPoint.x

                    val fittedDy =
                        fittedCentreY -
                                clickedPoint.y

                    val fittedClickDistance =
                        sqrt(
                            fittedDx *
                                    fittedDx +
                                    fittedDy *
                                    fittedDy
                        )

                    /*
                     * Click proximity becomes a strong part of the
                     * final score, but we still retain the actual
                     * distance separately for final selection.
                     */
                    val proximityScore =
                        (
                                1.0 -
                                        fittedClickDistance /
                                        (
                                                radius *
                                                        1.5 +
                                                        20.0
                                                )
                                )
                            .coerceIn(
                                0.0,
                                1.0
                            )

                    val score =
                        0.35 *
                                shapeScore +
                                0.25 *
                                boundary.coverage +
                                0.15 *
                                edgeScore +
                                0.10 *
                                fitScore +
                                0.15 *
                                proximityScore

                    if (
                        score <
                        MIN_FINAL_SCORE
                    ) {
                        return null
                    }

                    return RefinedCandidate(

                        detection =
                            BlobDetection(
                                centre =
                                    Point2(
                                        fittedCentreX
                                            .toFloat(),

                                        fittedCentreY
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

                        score =
                            score,

                        clickDistance =
                            fittedClickDistance
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
     * Find a boundary at the candidate's apparent scale.
     *
     * The previous detector searched a very wide radius and simply
     * selected the strongest gradient. That allowed shadows and
     * ground transitions to win.
     *
     * Here the expected radius comes from the scale at which the
     * candidate was detected, and gradients are weighted according
     * to their distance from that radius.
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

        val minRadius =
            (
                    expectedRadius *
                            MIN_EDGE_RADIUS_FRACTION
                    )
                .coerceAtLeast(5.0)

        val maxRadius =
            (
                    expectedRadius *
                            MAX_EDGE_RADIUS_FRACTION
                    )
                .coerceAtLeast(
                    minRadius + 5.0
                )

        val sigma =
            maxOf(
                1.0,
                expectedRadius *
                        EDGE_RADIUS_TOLERANCE
            )

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

            var bestRadius =
                Double.NaN

            var bestScore =
                0.0

            var bestGradient =
                0.0

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
                            (
                                    radius - 1.0
                                    )

                val previousY =
                    centre.y +
                            dy *
                            (
                                    radius - 1.0
                                    )

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

                /*
                 * Gaussian preference for the candidate scale.
                 *
                 * A very strong edge far from the expected object
                 * boundary therefore does not automatically win.
                 */
                val distanceFromExpected =
                    radius -
                            expectedRadius

                val scaleWeight =
                    exp(
                        -0.5 *
                                (
                                        distanceFromExpected /
                                                sigma
                                        ).pow(2)
                    )

                val weightedScore =
                    gradient *
                            scaleWeight

                if (
                    gradient >=
                    MIN_EDGE_STRENGTH &&
                    weightedScore >
                    bestScore
                ) {

                    bestScore =
                        weightedScore

                    bestGradient =
                        gradient

                    bestRadius =
                        radius
                }

                radius += 1.0
            }

            if (
                bestRadius.isFinite() &&
                bestGradient >=
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
                    bestGradient

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

            points =
                points,

            coverage =
                successful.toDouble() /
                        RADIAL_SAMPLES,

            meanStrength =
                strengthSum /
                        successful
        )
    }

    /**
     * Fit an ellipse to the boundary points.
     */
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

    /**
     * Measure how closely the boundary points lie on the ellipse.
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

            val normalized =
                sqrt(
                    ex * ex /
                            (a * a) +
                            ey * ey /
                            (b * b)
                )

            errorSum +=
                abs(
                    normalized - 1.0
                )
        }

        val meanError =
            errorSum /
                    points.size

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
     * Measure radial consistency against the fitted ellipse.
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
                                    b *
                                            cos(theta)
                                    ).pow(2) +
                                    (
                                            a *
                                                    sin(theta)
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
            }
        }

        val meanError =
            errorSum /
                    points.size

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
     * Convert Mat to greyscale.
     */
    private fun convertToGray(
        source: Mat,
        destination: Mat
    ) {

        when (
            source.channels()
        ) {

            4 ->
                Imgproc.cvtColor(
                    source,
                    destination,
                    Imgproc.COLOR_RGBA2GRAY
                )

            3 ->
                Imgproc.cvtColor(
                    source,
                    destination,
                    Imgproc.COLOR_RGB2GRAY
                )

            1 ->
                source.copyTo(
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
     * Sample a greyscale image at the nearest pixel.
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
     * Ensure a Gaussian kernel is odd and at least 3.
     */
    private fun makeOddKernel(
        value: Int
    ): Int {

        val minimum =
            maxOf(
                3,
                value
            )

        return if (
            minimum % 2 == 0
        ) {
            minimum + 1
        } else {
            minimum
        }
    }

    /**
     * Resize the complete image once.
     *
     * All candidate coordinates are working-image coordinates until
     * they are explicitly converted back to original-image pixels.
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
                            )
                        .roundToInt()
                        .toDouble(),

                    (
                            source.rows() *
                                    scale
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
            )
                .toDouble()

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

    private data class CandidateCentre(

        val centre: Point,

        val estimatedRadius: Double,

        val response: Double = 0.0
    )

    private data class BoundaryResult(

        val points: List<Point>,

        val coverage: Double,

        val meanStrength: Double
    )

    private data class RefinedCandidate(

        val detection: BlobDetection,

        val score: Double,

        val clickDistance: Double
    )
}