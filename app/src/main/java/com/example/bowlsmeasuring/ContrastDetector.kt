package com.example.bowlsmeasuring

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/**
 * Colour-blind wood detector.
 *
 * Deliberately does NOT use:
 *   - HSV
 *   - RGB colour thresholds
 *   - brown/red/green/blue classification
 *
 * Instead it looks for:
 *   - local luminance contrast
 *   - a substantial interior
 *   - plausible apparent size
 *   - compact geometry
 *   - approximately circular/elliptical shape
 *
 * Strongly elongated candidates are penalised because shadows beside
 * woods are commonly much more elongated than the wood itself.
 *
 * The detector returns at most four candidates.
 *
 * COORDINATE RULE:
 *
 * detectInRegion() works entirely in the coordinate system of the
 * Mat passed to it.
 *
 * detect() scales the complete image and converts the results back
 * to original bitmap coordinates.
 *
 * detectNear() scales the complete image, extracts an ROI, runs the
 * detector in ROI-local coordinates, then adds the ROI origin and
 * converts back to original bitmap coordinates.
 */
object ContrastWoodDetector {

    private const val TARGET_MAX_DIM = 1024.0

    /*
     * Apparent radius limits in the working image.
     */
    private const val MIN_RADIUS = 6.0
    private const val MAX_RADIUS = 100.0

    /*
     * Minimum luminance difference used to form the contrast mask.
     */
    private const val MIN_CONTRAST = 8.0

    /*
     * Maximum number of woods.
     */
    private const val MAX_DETECTIONS = 4

    /*
     * Local search size used by detectNear().
     */
    private const val NEAR_ROI_SIZE = 320

    /*
     * Candidates below this shape score are rejected.
     *
     * This is intentionally not especially aggressive because strong
     * perspective can make a genuine wood quite elliptical.
     */
    private const val MIN_SHAPE_SCORE = 0.30

    /**
     * Fully automatic detection.
     *
     * Returns coordinates in original bitmap pixels.
     */
    fun detect(
        bitmap: Bitmap
    ): List<BlobDetection> {

        val source = Mat()
        Utils.bitmapToMat(bitmap, source)

        if (source.empty()) {
            source.release()
            return emptyList()
        }

        try {

            /*
             * Resize the complete image once.
             */
            val scale =
                calculateScale(source)

            val working =
                resizeForDetection(
                    source,
                    scale
                )

            try {

                /*
                 * Detection is performed entirely in working-image
                 * coordinates.
                 */
                val candidates =
                    detectInRegion(
                        source = working,
                        tapPoint = null
                    )

                /*
                 * Convert working coordinates to original bitmap
                 * coordinates exactly once.
                 */
                return candidates.map {
                    scaleDetection(
                        detection = it,
                        scale = 1.0 / scale
                    )
                }

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
     * Localized detection.
     *
     * Searches around clickedPoint and returns the best candidate.
     *
     * clickedPoint and the returned detection are both in original
     * bitmap coordinates.
     */
    fun detectNear(
        bitmap: Bitmap,
        clickedPoint: Point2
    ): BlobDetection? {

        val source = Mat()
        Utils.bitmapToMat(bitmap, source)

        if (source.empty()) {
            source.release()
            return null
        }

        try {

            /*
             * ----------------------------------------------------------
             * 1. Resize the complete image ONCE.
             * ----------------------------------------------------------
             */

            val scale =
                calculateScale(source)

            val working =
                resizeForDetection(
                    source,
                    scale
                )

            try {

                /*
                 * ------------------------------------------------------
                 * 2. Convert click to working coordinates.
                 * ------------------------------------------------------
                 */

                val workingClick =
                    Point(
                        clickedPoint.x * scale,
                        clickedPoint.y * scale
                    )

                /*
                 * ------------------------------------------------------
                 * 3. Construct ROI in working-image coordinates.
                 * ------------------------------------------------------
                 */

                val half =
                    NEAR_ROI_SIZE / 2

                val sx =
                    maxOf(
                        0,
                        (workingClick.x - half)
                            .roundToInt()
                    )

                val sy =
                    maxOf(
                        0,
                        (workingClick.y - half)
                            .roundToInt()
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
                    width < 30 ||
                    height < 30
                ) {
                    return null
                }

                /*
                 * ------------------------------------------------------
                 * 4. Extract ROI.
                 *
                 * No further resizing occurs.
                 * ------------------------------------------------------
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

                    /*
                     * Click in ROI-local coordinates.
                     */
                    val roiClick =
                        Point(
                            workingClick.x - sx,
                            workingClick.y - sy
                        )

                    /*
                     * --------------------------------------------------
                     * 5. Detect in ROI-local coordinates.
                     * --------------------------------------------------
                     */

                    val candidates =
                        detectInRegion(
                            source = roi,
                            tapPoint = roiClick
                        )

                    if (
                        candidates.isEmpty()
                    ) {
                        return null
                    }

                    /*
                     * --------------------------------------------------
                     * 6. The detector has already scored proximity.
                     *
                     * candidates are sorted by score, so first is best.
                     * --------------------------------------------------
                     */

                    val best =
                        candidates.first()

                    /*
                     * --------------------------------------------------
                     * 7. ROI-local working coordinates
                     *          ->
                     *    complete working coordinates
                     *          ->
                     *    original bitmap coordinates
                     *
                     * This is the ONLY place the ROI origin is added.
                     * --------------------------------------------------
                     */

                    val completeWorking =
                        addOrigin(
                            detection = best,
                            originX = sx,
                            originY = sy
                        )

                    return scaleDetection(
                        detection = completeWorking,
                        scale = 1.0 / scale
                    )

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
     * Core detector.
     *
     * IMPORTANT:
     *
     * This function NEVER resizes the supplied Mat.
     *
     * Coordinates returned are relative to that Mat.
     */
    private fun detectInRegion(
        source: Mat,
        tapPoint: Point?
    ): List<BlobDetection> {

        val gray = Mat()
        val background = Mat()
        val contrast = Mat()
        val mask = Mat()
        val distance = Mat()
        val localMax = Mat()
        val peakMask = Mat()
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()

        try {

            /*
             * ----------------------------------------------------------
             * 1. Convert to luminance.
             * ----------------------------------------------------------
             */

            when (source.channels()) {

                4 -> Imgproc.cvtColor(
                    source,
                    gray,
                    Imgproc.COLOR_RGBA2GRAY
                )

                3 -> Imgproc.cvtColor(
                    source,
                    gray,
                    Imgproc.COLOR_RGB2GRAY
                )

                1 -> source.copyTo(gray)

                else -> return emptyList()
            }

            /*
             * Suppress fine grass texture and camera noise.
             */
            Imgproc.GaussianBlur(
                gray,
                gray,
                Size(5.0, 5.0),
                1.2
            )

            /*
             * ----------------------------------------------------------
             * 2. Local luminance contrast.
             *
             * This works for both black and brown woods.
             * ----------------------------------------------------------
             */

            Imgproc.GaussianBlur(
                gray,
                background,
                Size(31.0, 31.0),
                0.0
            )

            Core.absdiff(
                gray,
                background,
                contrast
            )

            Imgproc.threshold(
                contrast,
                mask,
                MIN_CONTRAST,
                255.0,
                Imgproc.THRESH_BINARY
            )

            /*
             * ----------------------------------------------------------
             * 3. Clean the contrast mask.
             * ----------------------------------------------------------
             */

            val openKernel =
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(5.0, 5.0)
                )

            val closeKernel =
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(11.0, 11.0)
                )

            try {

                Imgproc.morphologyEx(
                    mask,
                    mask,
                    Imgproc.MORPH_OPEN,
                    openKernel
                )

                Imgproc.morphologyEx(
                    mask,
                    mask,
                    Imgproc.MORPH_CLOSE,
                    closeKernel
                )

            } finally {

                openKernel.release()
                closeKernel.release()
            }

            /*
             * ----------------------------------------------------------
             * 4. Distance transform.
             *
             * Used primarily to find candidate centres.
             * It is NOT treated as evidence that something is a wood.
             * ----------------------------------------------------------
             */

            Imgproc.distanceTransform(
                mask,
                distance,
                Geometry.DIST_L2,
                5
            )

            /*
             * ----------------------------------------------------------
             * 5. Find local maxima.
             * ----------------------------------------------------------
             */

            val peakKernel =
                Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE,
                    Size(17.0, 17.0)
                )

            try {

                Imgproc.dilate(
                    distance,
                    localMax,
                    peakKernel
                )

            } finally {
                peakKernel.release()
            }

            Core.compare(
                distance,
                localMax,
                peakMask,
                Core.CMP_EQ
            )

            /*
             * Restrict peaks to plausible sizes.
             */
            val radiusMask =
                Mat()

            try {

                Core.inRange(
                    distance,
                    Scalar(MIN_RADIUS),
                    Scalar(MAX_RADIUS),
                    radiusMask
                )

                Core.bitwise_and(
                    peakMask,
                    radiusMask,
                    peakMask
                )

            } finally {
                radiusMask.release()
            }

            /*
             * ----------------------------------------------------------
             * 6. Group neighbouring maxima.
             * ----------------------------------------------------------
             */

            val componentCount =
                Imgproc.connectedComponentsWithStats(
                    peakMask,
                    labels,
                    stats,
                    centroids,
                    8,
                    CvType.CV_32S
                )

            val candidates =
                mutableListOf<ScoredCandidate>()

            /*
             * ----------------------------------------------------------
             * 7. Evaluate each candidate.
             * ----------------------------------------------------------
             */

            for (
            label in 1 until componentCount
            ) {

                val cx =
                    centroids.get(
                        label,
                        0
                    )[0]

                val cy =
                    centroids.get(
                        label,
                        1
                    )[0]

                if (
                    !cx.isFinite() ||
                    !cy.isFinite()
                ) {
                    continue
                }

                val ix =
                    cx.roundToInt()
                        .coerceIn(
                            0,
                            distance.cols() - 1
                        )

                val iy =
                    cy.roundToInt()
                        .coerceIn(
                            0,
                            distance.rows() - 1
                        )

                val radius =
                    distance
                        .get(iy, ix)
                        ?.firstOrNull()
                        ?: continue

                if (
                    radius < MIN_RADIUS ||
                    radius > MAX_RADIUS
                ) {
                    continue
                }

                /*
                 * ------------------------------------------------------
                 * 7a. Candidate mask.
                 * ------------------------------------------------------
                 */

                val candidateMask =
                    Mat.zeros(
                        mask.size(),
                        CvType.CV_8U
                    )

                try {

                    /*
                     * Use a region somewhat larger than the distance
                     * peak so that we can inspect the actual shape.
                     */
                    Imgproc.circle(
                        candidateMask,
                        Point(cx, cy),
                        maxOf(
                            1.0,
                            radius * 1.70
                        ).roundToInt(),
                        Scalar(255.0),
                        -1
                    )

                    Core.bitwise_and(
                        candidateMask,
                        mask,
                        candidateMask
                    )

                    /*
                     * --------------------------------------------------
                     * 7b. Candidate contour.
                     * --------------------------------------------------
                     */

                    val contours =
                        ArrayList<MatOfPoint>()

                    val hierarchy =
                        Mat()

                    try {

                        val contourInput =
                            candidateMask.clone()

                        try {

                            Imgproc.findContours(
                                contourInput,
                                contours,
                                hierarchy,
                                Imgproc.RETR_EXTERNAL,
                                Imgproc.CHAIN_APPROX_SIMPLE
                            )

                        } finally {
                            contourInput.release()
                        }

                        if (
                            contours.isEmpty()
                        ) {
                            continue
                        }

                        val contour =
                            contours.maxByOrNull {
                                Geometry.contourArea(it)
                            }
                                ?: continue

                        val contourArea =
                            Geometry.contourArea(
                                contour
                            )

                        if (
                            contourArea <= 0.0
                        ) {
                            continue
                        }

                        /*
                         * ------------------------------------------------
                         * 7c. Circularity.
                         *
                         * A shadow will usually score poorly here.
                         * ------------------------------------------------
                         */

                        val contour2f =
                            MatOfPoint2f(
                                *contour.toArray()
                            )

                        val perimeter =
                            try {

                                Geometry.arcLength(
                                    contour2f,
                                    true
                                )

                            } finally {
                                contour2f.release()
                            }

                        val circularity =
                            if (
                                perimeter > 0.0
                            ) {

                                (
                                        4.0 *
                                                PI *
                                                contourArea
                                        ) /
                                        (
                                                perimeter *
                                                        perimeter
                                                )

                            } else {
                                0.0
                            }

                        val circularityScore =
                            circularity
                                .coerceIn(
                                    0.0,
                                    1.0
                                )

                        /*
                         * ------------------------------------------------
                         * 7d. Ellipse geometry.
                         *
                         * IMPORTANT:
                         *
                         * A high aspect ratio is GOOD for a wood.
                         * A very low aspect ratio is BAD.
                         *
                         * We therefore use the aspect ratio as a
                         * progressive penalty rather than rewarding
                         * ellipse-ness.
                         * ------------------------------------------------
                         */

                        var aspectRatio =
                            0.0

                        if (
                            contour.total() >= 5
                        ) {

                            val contour2fEllipse =
                                MatOfPoint2f(
                                    *contour.toArray()
                                )

                            try {

                                val ellipse =
                                    Geometry.fitEllipse(
                                        contour2fEllipse
                                    )

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
                                    major > 0.0
                                ) {

                                    aspectRatio =
                                        (
                                                minor /
                                                        major
                                                ).coerceIn(
                                                0.0,
                                                1.0
                                            )
                                }

                            } finally {
                                contour2fEllipse.release()
                            }
                        }

                        /*
                         * ------------------------------------------------
                         * 7e. Shape score.
                         *
                         * Both circularity and aspect ratio matter.
                         *
                         * Examples:
                         *
                         *   round object:
                         *       circularity ~0.8
                         *       aspect       ~0.8
                         *
                         *   perspective-distorted wood:
                         *       circularity ~0.6
                         *       aspect       ~0.5
                         *
                         *   elongated shadow:
                         *       circularity ~0.3
                         *       aspect       ~0.2
                         * ------------------------------------------------
                         */

                        val shapeScore =
                            (
                                    0.60 *
                                            circularityScore +
                                            0.40 *
                                            aspectRatio
                                    ).coerceIn(
                                    0.0,
                                    1.0
                                )

                        /*
                         * Reject obviously elongated objects.
                         *
                         * We deliberately allow fairly severe
                         * perspective distortion.
                         */
                        if (
                            shapeScore <
                            MIN_SHAPE_SCORE
                        ) {
                            continue
                        }

                        /*
                         * ------------------------------------------------
                         * 7f. Fill ratio.
                         * ------------------------------------------------
                         */

                        val candidateArea =
                            Core.countNonZero(
                                candidateMask
                            ).toDouble()

                        val idealArea =
                            PI *
                                    radius *
                                    radius

                        val fillRatio =
                            (
                                    candidateArea /
                                            idealArea.coerceAtLeast(
                                                1.0
                                            )
                                    ).coerceIn(
                                    0.0,
                                    1.0
                                )

                        /*
                         * ------------------------------------------------
                         * 7g. Local interior/exterior contrast.
                         * ------------------------------------------------
                         */

                        val innerMask =
                            Mat.zeros(
                                mask.size(),
                                CvType.CV_8U
                            )

                        val outerMask =
                            Mat.zeros(
                                mask.size(),
                                CvType.CV_8U
                            )

                        val ringMask =
                            Mat.zeros(
                                mask.size(),
                                CvType.CV_8U
                            )

                        try {

                            Imgproc.circle(
                                innerMask,
                                Point(cx, cy),
                                maxOf(
                                    2.0,
                                    radius * 0.70
                                ).roundToInt(),
                                Scalar(255.0),
                                -1
                            )

                            Imgproc.circle(
                                outerMask,
                                Point(cx, cy),
                                maxOf(
                                    3.0,
                                    radius * 1.60
                                ).roundToInt(),
                                Scalar(255.0),
                                -1
                            )

                            Core.subtract(
                                outerMask,
                                innerMask,
                                ringMask
                            )

                            val innerMean =
                                Core.mean(
                                    gray,
                                    innerMask
                                ).`val`[0]

                            val outerMean =
                                Core.mean(
                                    gray,
                                    ringMask
                                ).`val`[0]

                            val localDifference =
                                abs(
                                    innerMean -
                                            outerMean
                                )

                            val contrastScore =
                                (
                                        localDifference /
                                                50.0
                                        ).coerceIn(
                                        0.0,
                                        1.0
                                    )

                            /*
                             * ------------------------------------------------
                             * 7h. Radius score.
                             *
                             * Don't make large objects automatically win.
                             * We just prefer candidates comfortably above
                             * the minimum detectable size.
                             * ------------------------------------------------
                             */

                            val radiusScore =
                                (
                                        (radius - MIN_RADIUS) /
                                                (
                                                        MAX_RADIUS -
                                                                MIN_RADIUS
                                                        )
                                        ).coerceIn(
                                        0.0,
                                        1.0
                                    )

                            /*
                             * ------------------------------------------------
                             * 7i. Tap proximity.
                             * ------------------------------------------------
                             */

                            val proximityScore =
                                if (
                                    tapPoint != null
                                ) {

                                    val dx =
                                        cx -
                                                tapPoint.x

                                    val dy =
                                        cy -
                                                tapPoint.y

                                    val distanceToTap =
                                        sqrt(
                                            dx * dx +
                                                    dy * dy
                                        )

                                    (
                                            1.0 -
                                                    distanceToTap /
                                                    160.0
                                            ).coerceIn(
                                            0.0,
                                            1.0
                                        )

                                } else {
                                    0.0
                                }

                            /*
                             * ------------------------------------------------
                             * 7j. Shadow/wood relationship.
                             *
                             * If there is an elongated contrast region
                             * immediately beside this candidate, that is
                             * actually useful evidence that this compact
                             * object may be the wood casting the shadow.
                             *
                             * This is handled later from the complete
                             * candidate list, because at this point we
                             * haven't evaluated all candidates yet.
                             * ------------------------------------------------
                             */

                            /*
                             * Shape deliberately dominates the score.
                             */
                            var score =
                                0.35 *
                                        shapeScore +
                                        0.25 *
                                        contrastScore +
                                        0.15 *
                                        fillRatio +
                                        0.10 *
                                        circularityScore +
                                        0.05 *
                                        aspectRatio +
                                        0.10 *
                                        radiusScore

                            if (
                                tapPoint != null
                            ) {

                                score =
                                    0.75 * score +
                                            0.25 *
                                            proximityScore
                            }

                            candidates +=
                                ScoredCandidate(
                                    detection =
                                        BlobDetection(
                                            centre =
                                                Point2(
                                                    cx.toFloat(),
                                                    cy.toFloat()
                                                ),

                                            radiusPx =
                                                radius.toFloat(),

                                            areaPixels =
                                                (
                                                        PI *
                                                                radius *
                                                                radius
                                                        ).roundToInt(),

                                            solidity =
                                                circularity
                                        ),

                                    score = score,

                                    x = cx,
                                    y = cy,

                                    radius = radius,

                                    circularity =
                                        circularityScore,

                                    aspectRatio =
                                        aspectRatio
                                )

                        } finally {

                            innerMask.release()
                            outerMask.release()
                            ringMask.release()
                        }

                    } finally {

                        hierarchy.release()

                        contours.forEach {
                            it.release()
                        }
                    }

                } finally {
                    candidateMask.release()
                }
            }

            /*
             * ----------------------------------------------------------
             * 8. Shadow association.
             *
             * A shadow can be easier for the contrast detector to see
             * than the wood itself.
             *
             * We therefore look for an elongated candidate near a
             * compact candidate and boost the compact candidate.
             *
             * This is NOT required for a candidate to be accepted.
             * ----------------------------------------------------------
             */

            for (
            i in candidates.indices
            ) {

                val candidate =
                    candidates[i]

                var shadowEvidence =
                    0.0

                for (
                j in candidates.indices
                ) {

                    if (i == j) {
                        continue
                    }

                    val possibleShadow =
                        candidates[j]

                    /*
                     * A shadow should be more elongated than the
                     * candidate we are evaluating.
                     */
                    if (
                        possibleShadow.aspectRatio >=
                        candidate.aspectRatio
                    ) {
                        continue
                    }

                    if (
                        possibleShadow.circularity >=
                        candidate.circularity
                    ) {
                        continue
                    }

                    val dx =
                        candidate.x -
                                possibleShadow.x

                    val dy =
                        candidate.y -
                                possibleShadow.y

                    val centreDistance =
                        sqrt(
                            dx * dx +
                                    dy * dy
                        )

                    /*
                     * Shadows should be nearby, but not necessarily
                     * overlapping.
                     */
                    val maximumDistance =
                        maxOf(
                            candidate.radius,
                            possibleShadow.radius
                        ) * 3.5

                    if (
                        centreDistance <=
                        maximumDistance
                    ) {

                        val elongationDifference =
                            (
                                    candidate.aspectRatio -
                                            possibleShadow.aspectRatio
                                    ).coerceIn(
                                    0.0,
                                    1.0
                                )

                        val circularityDifference =
                            (
                                    candidate.circularity -
                                            possibleShadow.circularity
                                    ).coerceIn(
                                    0.0,
                                    1.0
                                )

                        shadowEvidence =
                            maxOf(
                                shadowEvidence,
                                0.5 *
                                        elongationDifference +
                                        0.5 *
                                        circularityDifference
                            )
                    }
                }

                if (
                    shadowEvidence > 0.0
                ) {

                    /*
                     * Small boost only.
                     *
                     * We don't want the detector to require shadows,
                     * merely to use them as supporting evidence.
                     */
                    candidate.score +=
                        0.10 *
                                shadowEvidence
                }
            }

            /*
             * ----------------------------------------------------------
             * 9. Non-maximum suppression.
             * ----------------------------------------------------------
             */

            val selected =
                mutableListOf<ScoredCandidate>()

            for (
            candidate
            in candidates.sortedByDescending {
                it.score
            }
            ) {

                val overlapsExisting =
                    selected.any { existing ->

                        val dx =
                            candidate.x -
                                    existing.x

                        val dy =
                            candidate.y -
                                    existing.y

                        val centreDistance =
                            sqrt(
                                dx * dx +
                                        dy * dy
                            )

                        centreDistance <
                                maxOf(
                                    candidate.radius,
                                    existing.radius
                                ) * 1.25
                    }

                if (!overlapsExisting) {
                    selected += candidate
                }

                if (
                    selected.size >=
                    MAX_DETECTIONS
                ) {
                    break
                }
            }

            /*
             * ----------------------------------------------------------
             * 10. Return highest scoring candidates.
             *
             * Coordinates are still relative to the supplied Mat.
             * ----------------------------------------------------------
             */

            return selected
                .sortedByDescending {
                    it.score
                }
                .map {
                    it.detection
                }

        } catch (_: Exception) {

            return emptyList()

        } finally {

            gray.release()
            background.release()
            contrast.release()
            mask.release()
            distance.release()
            localMax.release()
            peakMask.release()
            labels.release()
            stats.release()
            centroids.release()
        }
    }

    /**
     * Resize a source image for detection.
     */
    private fun resizeForDetection(
        source: Mat,
        scale: Double
    ): Mat {

        val result =
            Mat()

        if (scale == 1.0) {

            source.copyTo(result)

        } else {

            Imgproc.resize(
                source,
                result,
                Size(
                    (
                            source.cols() *
                                    scale
                            ).roundToInt().toDouble(),

                    (
                            source.rows() *
                                    scale
                            ).roundToInt().toDouble()
                )
            )
        }

        return result
    }

    /**
     * Add an ROI origin to a detection.
     *
     * Input and output coordinates are both in working-image pixels.
     */
    private fun addOrigin(
        detection: BlobDetection,
        originX: Int,
        originY: Int
    ): BlobDetection {

        return BlobDetection(

            centre =
                Point2(
                    (
                            detection.centre.x +
                                    originX
                            ).toFloat(),

                    (
                            detection.centre.y +
                                    originY
                            ).toFloat()
                ),

            radiusPx =
                detection.radiusPx,

            areaPixels =
                detection.areaPixels,

            solidity =
                detection.solidity
        )
    }

    /**
     * Scale a detection between coordinate systems.
     *
     * scale = originalPixels / workingPixels
     *
     * Therefore:
     *
     *   coordinate *= scale
     *   radius     *= scale
     *   area       *= scale²
     */
    private fun scaleDetection(
        detection: BlobDetection,
        scale: Double
    ): BlobDetection {

        return BlobDetection(

            centre =
                Point2(
                    (
                            detection.centre.x *
                                    scale
                            ).toFloat(),

                    (
                            detection.centre.y *
                                    scale
                            ).toFloat()
                ),

            radiusPx =
                (
                        detection.radiusPx *
                                scale
                        ).toFloat(),

            areaPixels =
                (
                        detection.areaPixels *
                                scale *
                                scale
                        ).roundToInt(),

            solidity =
                detection.solidity
        )
    }

    private fun calculateScale(
        source: Mat
    ): Double {

        val maxDimension =
            maxOf(
                source.cols(),
                source.rows()
            ).toDouble()

        return if (
            maxDimension >
            TARGET_MAX_DIM
        ) {

            TARGET_MAX_DIM /
                    maxDimension

        } else {

            1.0
        }
    }

    private data class ScoredCandidate(

        val detection: BlobDetection,

        var score: Double,

        val x: Double,

        val y: Double,

        val radius: Double,

        val circularity: Double,

        val aspectRatio: Double
    )
}

