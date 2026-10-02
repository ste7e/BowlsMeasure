package com.example.bowlsmeasuring

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

object ObjectDetector {

    private const val MAX_WORKING_SIZE = 1024

    /*
     * Search area around the user's click.
     */
    private const val ROI_SIZE = 360

    /*
     * Radial boundary search.
     */
    private const val ANGLE_SAMPLES = 144
    private const val MIN_RADIUS = 10.0
    private const val MAX_RADIUS = 110.0
    private const val RADIUS_STEP = 1.0
    /*
 * Boundary evidence.
 *
 * These are deliberately fairly permissive initially. We want to
 * diagnose the detector before tightening the thresholds.
 */
    private const val BOUNDARY_POINT_THRESHOLD = 0.15

    /*
     * Minimum fraction of the circumference that should contain coherent
     * boundary evidence.
     */
    private const val MIN_COHERENT_RUN = 0.08

    /*
     * Candidate confidence threshold.
     *
     * This is deliberately lower than "certain". The centre-search stage
     * will still compare candidates against one another.
     */
    private const val MIN_CONFIDENCE = 0.45

    /*
     * The centre is allowed to move slightly away from
     * the user's click.
     */
    private const val CENTRE_SEARCH_RADIUS = 20.0
    private const val CENTRE_SEARCH_STEP = 5.0

    /*
     * A boundary should be supported by a reasonable
     * proportion of the radial directions.
     */
    private const val MIN_BOUNDARY_COVERAGE = 0.45
    private const val MIN_OPPOSITE_COHERENCE = 0.52

    private const val MIN_CIRCULAR_CONTRAST = 0.18
    private const val MIN_CONTRAST_COVERAGE = 0.55
    private const val CONTRAST_SAMPLE_INNER = 0.80
    private const val CONTRAST_SAMPLE_OUTER = 1.20

    private const val TAG = "ObjectDetector"
    private const val DIAGNOSTIC = true

    /*
     * Ignore isolated strong edges. A genuine object
     * boundary should produce a reasonably coherent edge./*
 * ------------------------------------------------------------
 * Opposite-side boundary coherence
 * ------------------------------------------------------------
 *
 * A genuine circular object should have boundary evidence
 * on opposite sides of the circle.
 *
 * Random grass texture or a shadow edge may produce plenty
 * of individual edge points, but is much less likely to
 * produce corresponding evidence at theta and theta + PI.
 */
val oppositePairs = mutableListOf<Double>()

for (angleIndex in 0 until ANGLE_SAMPLES / 2) {

    val oppositeIndex =
        angleIndex + ANGLE_SAMPLES / 2

    val a =
        boundaryPoints[angleIndex].strength

    val b =
        boundaryPoints[oppositeIndex].strength

    /*
     * Both sides need to have useful evidence.
     * The weaker side determines the pair strength.
     */
    oppositePairs.add(
        min(a, b)
    )
}

val oppositeCoherence =
    oppositePairs.count {
        it >= MIN_BOUNDARY_POINT_STRENGTH
    }.toDouble() /
            oppositePairs.size

if (
    oppositeCoherence <
    MIN_OPPOSITE_COHERENCE
) {
    return null
}
     */
    private const val MIN_EDGE_STRENGTH = 12.0
    private const val MIN_APPEARANCE_DROP = 0.20
    private const val MIN_BOUNDARY_POINT_STRENGTH = 0.25
    private const val MIN_VALID_BOUNDARY_POINTS = 24
    data class BlobDetection(
        val centre: Point2,
        val radiusPx: Float,
        val areaPixels: Int,
        val solidity: Float
    )

    private data class BoundaryPoint(
        val angle: Double,
        val radius: Double,
        val strength: Double
    )

    private data class BoundaryResult(
        val centre: Point,
        val radius: Double,
        val points: List<BoundaryPoint>,
        val coverage: Double,
        val consistency: Double,
//        val coherence: Double,
        val interiorSupport: Double,
        val separation: Double,
        val fitQuality: Double,
        val score: Double,
//        val oppositeCoherence: Double,
    )
    fun detectNear(
        bitmap: Bitmap,
        clickedPoint: Point2
    ): BlobDetection? {

        val source = Mat()

        try {
            Utils.bitmapToMat(bitmap, source)

            val scale =
                min(
                    1.0,
                    MAX_WORKING_SIZE.toDouble() /
                            max(
                                source.cols(),
                                source.rows()
                            ).toDouble()
                )

            val working = Mat()

            try {

                if (scale < 0.999) {
                    Imgproc.resize(
                        source,
                        working,
                        Size(),
                        scale,
                        scale,
                        Imgproc.INTER_AREA
                    )
                } else {
                    source.copyTo(working)
                }

                val click = Point(
                    clickedPoint.x * scale,
                    clickedPoint.y * scale
                )

                val roiRect =
                    makeRoi(
                        click,
                        working.cols(),
                        working.rows()
                    )

                val roi =
                    Mat(
                        working,
                        roiRect
                    )

                try {

                    val localClick =
                        Point(
                            click.x - roiRect.x,
                            click.y - roiRect.y
                        )

                    val result =
                        findBoundary(
                            roi,
                            localClick
                        ) ?: return null

                    /*
                     * Convert the fitted boundary into the
                     * application's BlobDetection.
                     */
                    return makeDetection(
                        result,
                        scale,
                        roiRect
                    )

                } finally {
                    roi.release()
                }

            } finally {
                working.release()
            }

        } finally {
            source.release()
        }
    }

    private fun findBoundary(
        roi: Mat,
        click: Point
    ): BoundaryResult? {

        val gray = Mat()

        Imgproc.cvtColor(
            roi,
            gray,
            Imgproc.COLOR_RGBA2GRAY
        )

        val gradientX = Mat()
        val gradientY = Mat()
        val magnitude = Mat()

        Imgproc.Sobel(
            gray,
            gradientX,
            CvType.CV_32F,
            1,
            0
        )

        Imgproc.Sobel(
            gray,
            gradientY,
            CvType.CV_32F,
            0,
            1
        )

        Core.magnitude(
            gradientX,
            gradientY,
            magnitude
        )

        /*
         * Keep every candidate that passes findBoundaryForCentre().
         *
         * We will sort these afterwards so that Logcat shows us
         * exactly what the detector thinks are the best candidates.
         */
        val candidates =
            mutableListOf<BoundaryResult>()

        val searchRadius = 10.0
        val searchStep = 2.0

        var dy = -searchRadius

        while (dy <= searchRadius) {

            var dx = -searchRadius

            while (dx <= searchRadius) {

                if (
                    dx * dx +
                    dy * dy <=
                    searchRadius * searchRadius
                ) {

                    val candidateCentre =
                        Point(
                            click.x + dx,
                            click.y + dy
                        )

                    val candidate =
                        findBoundaryForCentre(
                            gray,
                            magnitude,
                            candidateCentre
                        )

                    if (candidate != null) {

                        /*
                         * The click MUST be inside the fitted circle.
                         */
                        val clickDistance =
                            hypot(
                                candidate.centre.x -
                                        click.x,

                                candidate.centre.y -
                                        click.y
                            )

                        if (
                            clickDistance <=
                            candidate.radius
                        ) {

                            candidates.add(candidate)

                            Log.d(
                                TAG,
                                String.format(
                                    java.util.Locale.US,
                                    "candidate centre=(%.1f,%.1f) " +
                                            "fitCentre=(%.1f,%.1f) " +
                                            "r=%.1f " +
                                            "score=%.3f " +
                                            "coverage=%.3f " +
                                            "consistency=%.3f " +
                                            "interior=%.3f " +
                                            "separation=%.3f",
                                    candidateCentre.x,
                                    candidateCentre.y,
                                    candidate.centre.x,
                                    candidate.centre.y,
                                    candidate.radius,
                                    candidate.score,
                                    candidate.coverage,
                                    candidate.consistency,
                                    candidate.interiorSupport,
                                    candidate.separation
                                )
                            )
                        }
                    }
                }

                dx += searchStep
            }

            dy += searchStep
        }

        /*
         * Sort exactly as the detector currently chooses the winner.
         */
        val best =
            candidates.maxByOrNull {
                it.score
            }

        /*
         * Print a compact summary of the top five candidates.
         */
        Log.d(
            TAG,
            "----------------------------------------"
        )

        Log.d(
            TAG,
            String.format(
                java.util.Locale.US,
                "CLICK x=%.1f y=%.1f candidates=%d",
                click.x,
                click.y,
                candidates.size
            )
        )

        candidates
            .sortedByDescending {
                it.score
            }
            .take(5)
            .forEachIndexed { index, candidate ->

                Log.d(
                    TAG,
                    String.format(
                        java.util.Locale.US,
                        "TOP %d: centre=(%.1f,%.1f) " +
                                "radius=%.1f " +
                                "score=%.3f " +
                                "coverage=%.3f " +
                                "consistency=%.3f " +
                                "interior=%.3f " +
                                "separation=%.3f",
                        index + 1,
                        candidate.centre.x,
                        candidate.centre.y,
                        candidate.radius,
                        candidate.score,
                        candidate.coverage,
                        candidate.consistency,
                        candidate.interiorSupport,
                        candidate.separation
                    )
                )
            }

        if (best != null) {

            Log.d(
                TAG,
                String.format(
                    java.util.Locale.US,
                    "WINNER: centre=(%.1f,%.1f) " +
                            "radius=%.1f score=%.3f",
                    best.centre.x,
                    best.centre.y,
                    best.radius,
                    best.score
                )
            )

        } else {

            Log.d(
                TAG,
                "WINNER: NONE"
            )
        }

        Log.d(
            TAG,
            "----------------------------------------"
        )

        gray.release()
        gradientX.release()
        gradientY.release()
        magnitude.release()

        return best
    }

    private data class RadiusResult(
        val radius: Double,
        val support: Double,
        val gradientScore: Double,
        val score: Double
    )

    private fun findBoundaryForCentre(
        gray: Mat,
        gradient: Mat,
        centre: Point
    ): BoundaryResult? {

        fun reject(
            reason: String,
            details: String = ""
        ): BoundaryResult? {

            if (DIAGNOSTIC) {
                Log.d(
                    TAG,
                    String.format(
                        java.util.Locale.US,
                        "REJECT centre=(%.1f,%.1f) reason=%s %s",
                        centre.x,
                        centre.y,
                        reason,
                        details
                    )
                )
            }

            return null
        }


        /*
         * ------------------------------------------------------------
         * 1. Establish seed appearance
         * ------------------------------------------------------------
         */

        val seedRadius = 8.0

        val seedValues =
            mutableListOf<Double>()

        for (dy in -seedRadius.toInt()..seedRadius.toInt()) {

            for (dx in -seedRadius.toInt()..seedRadius.toInt()) {

                if (
                    dx * dx +
                    dy * dy <=
                    seedRadius * seedRadius
                ) {

                    val value =
                        sampleGray(
                            gray,
                            centre.x + dx,
                            centre.y + dy
                        )

                    if (value != null) {
                        seedValues.add(value)
                    }
                }
            }
        }

        if (seedValues.size < 10) {

            return reject(
                "SEED",
                "samples=${seedValues.size}"
            )
        }

        seedValues.sort()

        val seedMedian =
            seedValues[
                seedValues.size / 2
            ]

        val seedMean =
            seedValues.average()

        val seedDeviation =
            sqrt(
                seedValues
                    .map {
                        val d =
                            it - seedMean

                        d * d
                    }
                    .average()
            )

        val appearanceTolerance =
            max(
                12.0,
                seedDeviation * 2.5
            )


        /*
         * ------------------------------------------------------------
         * 2. Measure radial appearance support
         * ------------------------------------------------------------
         */

        val radiusCount =
            (
                    (MAX_RADIUS - MIN_RADIUS) /
                            RADIUS_STEP
                    ).toInt() + 1

        val ringSupport =
            Array(
                radiusCount
            ) {
                DoubleArray(
                    ANGLE_SAMPLES
                )
            }

        val ringGradient =
            Array(
                radiusCount
            ) {
                DoubleArray(
                    ANGLE_SAMPLES
                )
            }

        for (radiusIndex in 0 until radiusCount) {

            val radius =
                MIN_RADIUS +
                        radiusIndex *
                        RADIUS_STEP

            for (angleIndex in 0 until ANGLE_SAMPLES) {

                val angle =
                    2.0 * Math.PI *
                            angleIndex /
                            ANGLE_SAMPLES

                val x =
                    centre.x +
                            radius *
                            cos(angle)

                val y =
                    centre.y +
                            radius *
                            sin(angle)

                val grayValue =
                    sampleGray(
                        gray,
                        x,
                        y
                    )

                val gradientValue =
                    sampleGradient(
                        gradient,
                        x,
                        y
                    )

                if (
                    grayValue != null &&
                    gradientValue != null
                ) {

                    val difference =
                        abs(
                            grayValue -
                                    seedMedian
                        )

                    val appearance =
                        max(
                            0.0,
                            1.0 -
                                    difference /
                                    appearanceTolerance
                        )

                    ringSupport[
                        radiusIndex
                    ][
                        angleIndex
                    ] =
                        appearance

                    ringGradient[
                        radiusIndex
                    ][
                        angleIndex
                    ] =
                        gradientValue
                }
            }
        }


        /*
         * ------------------------------------------------------------
         * 3. Score each radius
         * ------------------------------------------------------------
         */

        val radiusResults =
            mutableListOf<RadiusResult>()

        for (radiusIndex in 0 until radiusCount) {

            val radius =
                MIN_RADIUS +
                        radiusIndex *
                        RADIUS_STEP

            val support =
                ringSupport[
                    radiusIndex
                ].average()

            val gradientScore =
                min(
                    1.0,
                    ringGradient[
                        radiusIndex
                    ].average() / 80.0
                )

            val previousSupport =
                if (radiusIndex > 0) {

                    ringSupport[
                        radiusIndex - 1
                    ].average()

                } else {
                    support
                }

            val appearanceDrop =
                max(
                    0.0,
                    previousSupport -
                            support
                )

            val score =
                appearanceDrop * 0.60 +
                        gradientScore * 0.20 +
                        support * 0.20

            radiusResults.add(
                RadiusResult(
                    radius = radius,
                    support = support,
                    gradientScore = gradientScore,
                    score = score
                )
            )
        }


        /*
         * ------------------------------------------------------------
         * 4. Find local maxima
         * ------------------------------------------------------------
         */

        val peaks =
            mutableListOf<RadiusResult>()

        for (i in 1 until radiusResults.lastIndex) {

            val previous =
                radiusResults[i - 1]

            val current =
                radiusResults[i]

            val next =
                radiusResults[i + 1]

            if (
                current.score >= previous.score &&
                current.score >= next.score
            ) {

                peaks.add(current)
            }
        }

        if (peaks.isEmpty()) {

            return reject(
                "NO_PEAK"
            )
        }

        if (DIAGNOSTIC) {
            val peakText = peaks
                .sortedBy { it.radius }
                .joinToString(" | ") {
                    "r=${"%.1f".format(it.radius)} " +
                            "score=${"%.3f".format(it.score)} " +
                            "support=${"%.3f".format(it.support)} " +
                            "grad=${"%.3f".format(it.gradientScore)}"
                }

            Log.d(
                TAG,
                "PEAKS centre=(${"%.1f".format(centre.x)},${"%.1f".format(centre.y)}): $peakText"
            )
        }

        /*
         * ------------------------------------------------------------
         * 5. Select first strong boundary
         * ------------------------------------------------------------
         */

        val strongestScore =
            peaks.maxOf {
                it.score
            }

        val minimumPeakScore =
            strongestScore * 0.45

        val selected =
            peaks
                .filter {
                    it.score >=
                            minimumPeakScore
                }
                .minByOrNull {
                    it.radius
                }
                ?: return reject(
                    "NO_SELECTED_PEAK"
                )


        /*
         * Diagnostic: report the selected radius.
         */
        if (DIAGNOSTIC) {

            Log.d(
                TAG,
                String.format(
                    java.util.Locale.US,
                    "SELECT centre=(%.1f,%.1f) " +
                            "radius=%.1f " +
                            "score=%.3f " +
                            "support=%.3f " +
                            "gradient=%.3f " +
                            "strongest=%.3f",
                    centre.x,
                    centre.y,
                    selected.radius,
                    selected.score,
                    selected.support,
                    selected.gradientScore,
                    strongestScore
                )
            )
        }


        /*
         * ------------------------------------------------------------
         * 6. Interior support
         * ------------------------------------------------------------
         */

        val interiorFractions =
            doubleArrayOf(
                0.20,
                0.35,
                0.50,
                0.65,
                0.80
            )

        val interiorSupports =
            mutableListOf<Double>()

        for (fraction in interiorFractions) {

            val radius =
                selected.radius *
                        fraction

            val radiusIndex =
                (
                        (radius - MIN_RADIUS) /
                                RADIUS_STEP
                        )
                    .roundToInt()
                    .coerceIn(
                        0,
                        radiusCount - 1
                    )

            interiorSupports.add(
                ringSupport[
                    radiusIndex
                ].average()
            )
        }

        val interiorSupport =
            interiorSupports.average()


        /*
         * ------------------------------------------------------------
         * 7. Outside support
         * ------------------------------------------------------------
         */

        val outsideSupport =
            calculateOutsideSupport(
                gray,
                centre,
                selected.radius,
                seedMedian,
                appearanceTolerance
            )


        /*
         * ------------------------------------------------------------
         * 8. Boundary separation
         * ------------------------------------------------------------
         */

        val separation =
            max(
                0.0,
                interiorSupport -
                        outsideSupport
            )

        val separationScore =
            min(
                1.0,
                separation / 0.50
            )


        /*
         * ------------------------------------------------------------
         * 9. Recover boundary points
         * ------------------------------------------------------------
         */

        val boundaryPoints =
            mutableListOf<BoundaryPoint>()

        val selectedIndex =
            (
                    (selected.radius - MIN_RADIUS) /
                            RADIUS_STEP
                    )
                .roundToInt()
                .coerceIn(
                    1,
                    radiusCount - 2
                )

        for (angleIndex in 0 until ANGLE_SAMPLES) {

            var bestRadius =
                selected.radius

            var bestStrength =
                0.0

            for (offset in -4..4) {

                val radiusIndex =
                    (
                            selectedIndex +
                                    offset
                            )
                        .coerceIn(
                            1,
                            radiusCount - 2
                        )

                val insideSupport =
                    ringSupport[
                        radiusIndex - 1
                    ][
                        angleIndex
                    ]

                val outsideSupportAtPoint =
                    ringSupport[
                        radiusIndex + 1
                    ][
                        angleIndex
                    ]

                val appearanceDrop =
                    max(
                        0.0,
                        insideSupport -
                                outsideSupportAtPoint
                    )

                val gradientStrength =
                    ringGradient[
                        radiusIndex
                    ][
                        angleIndex
                    ]

                val strength =
                    appearanceDrop * 0.75 +
                            min(
                                1.0,
                                gradientStrength / 80.0
                            ) * 0.25

                if (
                    strength >
                    bestStrength
                ) {

                    bestStrength =
                        strength

                    bestRadius =
                        MIN_RADIUS +
                                radiusIndex *
                                RADIUS_STEP
                }
            }

            boundaryPoints.add(
                BoundaryPoint(
                    angle =
                        2.0 * Math.PI *
                                angleIndex /
                                ANGLE_SAMPLES,

                    radius =
                        bestRadius,

                    strength =
                        bestStrength
                )
            )
        }


        /*
         * ------------------------------------------------------------
         * 10. Boundary coverage
         * ------------------------------------------------------------
         */

        val coverage =
            boundaryPoints.count {
                it.strength >=
                        0.15
            }.toDouble() /
                    boundaryPoints.size


        /*
         * ------------------------------------------------------------
         * 11. Convert to Cartesian points
         * ------------------------------------------------------------
         */

        val cartesianPoints =
            boundaryPoints.map {

                Point(
                    centre.x +
                            it.radius *
                            cos(it.angle),

                    centre.y +
                            it.radius *
                            sin(it.angle)
                )
            }


        /*
         * ------------------------------------------------------------
         * 12. Fit circle
         * ------------------------------------------------------------ */

        val fitted =
            fitCircle(
                cartesianPoints
            )
                ?: return reject(
                    "FIT_CIRCLE",
                    String.format(
                        java.util.Locale.US,
                        "radius=%.1f coverage=%.3f",
                        selected.radius,
                        coverage
                    )
                )


        /*
         * ------------------------------------------------------------
         * 13. RMS rejection
         * ------------------------------------------------------------
         */

        val maximumRmsError =
            max(
                6.0,
                fitted.radius * 0.10
            )

        if (
            fitted.rmsError >
            maximumRmsError
        ) {

            return reject(
                "RMS",
                String.format(
                    java.util.Locale.US,
                    "radius=%.1f rms=%.2f max=%.2f " +
                            "coverage=%.3f",
                    fitted.radius,
                    fitted.rmsError,
                    maximumRmsError,
                    coverage
                )
            )
        }


        /*
         * ------------------------------------------------------------
         * 14. Click containment
         * ------------------------------------------------------------
         */

        val clickDistance =
            hypot(
                fitted.centre.x -
                        centre.x,

                fitted.centre.y -
                        centre.y
            )

        if (
            clickDistance >
            fitted.radius
        ) {

            return reject(
                "CLICK_OUTSIDE",
                String.format(
                    java.util.Locale.US,
                    "fitRadius=%.1f clickDistance=%.1f",
                    fitted.radius,
                    clickDistance
                )
            )
        }


        /*
         * ------------------------------------------------------------
         * 15. Boundary consistency
         * ------------------------------------------------------------ */

        val radiusMean =
            boundaryPoints
                .map {
                    it.radius
                }
                .average()

        val radiusVariance =
            boundaryPoints
                .map {

                    val d =
                        it.radius -
                                radiusMean

                    d * d

                }
                .average()

        val radiusStdDev =
            sqrt(
                radiusVariance
            )

        /*
         * ------------------------------------------------------------
         * Boundary consistency
         * ------------------------------------------------------------
         *
         * Measure how much the recovered boundary radius varies
         * relative to the actual object size.
         *
         * The previous 15% scale was too aggressive for small
         * objects: a ~2 px variation on a 13 px object was being
         * treated as almost completely inconsistent.
         *
         * Use 30% of the radius as the tolerance instead.
         */
        val consistencyTolerance =
            max(
                2.0,
                selected.radius * 0.30
            )

        val consistency =
            1.0 -
                    min(
                        1.0,
                        radiusStdDev /
                                consistencyTolerance
                    )
        /*
         * ------------------------------------------------------------
         * 16. Edge strength
         * ------------------------------------------------------------
         */

        val edgeScore =
            boundaryPoints
                .map {
                    it.strength
                }
                .average()


        /*
         * ------------------------------------------------------------
         * 17. Current hard validity tests
         * ------------------------------------------------------------
         */

        if (consistency < 0.20) {

            return reject(
                "CONSISTENCY",
                String.format(
                    java.util.Locale.US,
                    "radius=%.1f std=%.2f consistency=%.3f " +
                            "coverage=%.3f separation=%.3f",
                    fitted.radius,
                    radiusStdDev,
                    consistency,
                    coverage,
                    separationScore
                )
            )
        }

        if (separationScore < 0.10) {

            return reject(
                "SEPARATION",
                String.format(
                    java.util.Locale.US,
                    "radius=%.1f separation=%.3f " +
                            "interior=%.3f outside=%.3f " +
                            "coverage=%.3f consistency=%.3f",
                    fitted.radius,
                    separationScore,
                    interiorSupport,
                    outsideSupport,
                    coverage,
                    consistency
                )
            )
        }


        /*
         * ------------------------------------------------------------
         * 18. Final score
         * ------------------------------------------------------------
         */

        val finalScore =
            selected.score * 0.30 +
                    interiorSupport * 0.20 +
                    separationScore * 0.15 +
                    coverage * 0.15 +
                    consistency * 0.10 +
                    edgeScore * 0.10


        if (DIAGNOSTIC) {

            Log.d(
                TAG,
                String.format(
                    java.util.Locale.US,
                    "ACCEPT centre=(%.1f,%.1f) " +
                            "radius=%.1f score=%.3f " +
                            "coverage=%.3f consistency=%.3f " +
                            "interior=%.3f separation=%.3f " +
                            "edge=%.3f rms=%.2f",
                    fitted.centre.x,
                    fitted.centre.y,
                    fitted.radius,
                    finalScore,
                    coverage,
                    consistency,
                    interiorSupport,
                    separationScore,
                    edgeScore,
                    fitted.rmsError
                )
            )
        }


        return BoundaryResult(

            centre =
                fitted.centre,

            radius =
                fitted.radius,

            points =
                boundaryPoints,

            coverage =
                coverage,

            consistency =
                consistency,

            interiorSupport =
                interiorSupport,

            separation =
                separationScore,

            fitQuality =
                min(
                    1.0,
                    1.0 -
                            fitted.rmsError /
                            maximumRmsError
                ),

            score =
                finalScore
        )
    }


    /**
     * Measures whether boundary evidence is distributed coherently around
     * the circumference rather than appearing as isolated patches.
     *
     * A grass shadow can produce several strong edges which happen to fit
     * a circle. A real object boundary should normally have longer runs of
     * supporting evidence.
     */
    private fun calculateBoundaryCoherence(
        points: List<BoundaryPoint>
    ): Double {

        if (points.isEmpty()) {
            return 0.0
        }

        val supported =
            points.map {
                it.strength >= BOUNDARY_POINT_THRESHOLD
            }

        val count = supported.size

        if (count < 8) {
            return 0.0
        }

        /*
         * Find the longest continuous run of supported points.
         *
         * The boundary is circular, so the run may cross index 0.
         */
        var longestRun = 0
        var currentRun = 0

        for (i in 0 until count * 2) {

            if (supported[i % count]) {
                currentRun++

                if (currentRun > longestRun) {
                    longestRun = currentRun
                }

                /*
                 * Don't allow a completely supported circle to count twice.
                 */
                if (currentRun >= count) {
                    longestRun = count
                    break
                }

            } else {
                currentRun = 0
            }
        }

        val longestFraction =
            longestRun.toDouble() / count.toDouble()

        /*
         * Also consider the number of supported points.
         *
         * This stops one very long but extremely thin section from being
         * considered sufficient evidence on its own.
         */
        val coverage =
            supported.count { it }.toDouble() / count.toDouble()

        /*
         * Geometric mean rewards candidates where both properties are good.
         */
        return sqrt(
            longestFraction.coerceIn(0.0, 1.0) *
                    coverage.coerceIn(0.0, 1.0)
        )
    }

    private data class FittedCircle(
        val centre: Point,
        val radius: Double,
        val rmsError: Double
    )

    private fun fitCircle(
        points: List<Point>
    ): FittedCircle? {

        if (points.size < 8) {
            return null
        }

        /*
         * Algebraic least-squares circle fit.
         *
         * Circle:
         *
         *   x² + y² + D x + E y + F = 0
         *
         * Centre:
         *
         *   (-D/2, -E/2)
         */

        var sumX = 0.0
        var sumY = 0.0
        var sumX2 = 0.0
        var sumY2 = 0.0
        var sumXY = 0.0
        var sumX3 = 0.0
        var sumY3 = 0.0
        var sumX2Y = 0.0
        var sumXY2 = 0.0

        for (point in points) {

            val x = point.x
            val y = point.y

            val x2 = x * x
            val y2 = y * y

            sumX += x
            sumY += y
            sumX2 += x2
            sumY2 += y2
            sumXY += x * y

            sumX3 += x2 * x
            sumY3 += y2 * y
            sumX2Y += x2 * y
            sumXY2 += x * y2
        }

        val n =
            points.size.toDouble()

        /*
         * Solve:
         *
         * [Σx²  Σxy  Σx] [D] = -[Σx³ + Σxy²]
         * [Σxy  Σy²  Σy] [E]   -[Σx²y + Σy³]
         * [Σx   Σy    n ] [F]   -[Σx² + Σy²]
         */

        val matrix = Mat(3, 3, CvType.CV_64F)
        val rhs = Mat(3, 1, CvType.CV_64F)
        val solution = Mat()

        matrix.put(
            0, 0,
            sumX2,
            sumXY,
            sumX,
            sumXY,
            sumY2,
            sumY,
            sumX,
            sumY,
            n
        )

        rhs.put(
            0, 0,
            -(
                    sumX3 +
                            sumXY2
                    ),
            -(
                    sumX2Y +
                            sumY3
                    ),
            -(
                    sumX2 +
                            sumY2
                    )
        )

        val solved =
            Core.solve(
                matrix,
                rhs,
                solution,
                Core.DECOMP_SVD
            )

        if (!solved) {
            matrix.release()
            rhs.release()
            solution.release()
            return null
        }

        val d =
            solution.get(0, 0)[0]

        val e =
            solution.get(1, 0)[0]

        val f =
            solution.get(2, 0)[0]

        matrix.release()
        rhs.release()
        solution.release()

        val centreX =
            -d / 2.0

        val centreY =
            -e / 2.0

        val radiusSquared =
            centreX * centreX +
                    centreY * centreY -
                    f

        if (
            radiusSquared <= 0.0 ||
            !radiusSquared.isFinite()
        ) {
            return null
        }

        val radius =
            sqrt(radiusSquared)

        /*
         * Calculate geometric RMS error.
         */

        var squaredError = 0.0

        for (point in points) {

            val dx =
                point.x -
                        centreX

            val dy =
                point.y -
                        centreY

            val distance =
                sqrt(
                    dx * dx +
                            dy * dy
                )

            val error =
                distance -
                        radius

            squaredError +=
                error * error
        }

        val rmsError =
            sqrt(
                squaredError /
                        n
            )

        return FittedCircle(
            centre =
                Point(
                    centreX,
                    centreY
                ),
            radius = radius,
            rmsError = rmsError
        )
    }

    private fun calculateOutsideSupport(
        gray: Mat,
        centre: Point,
        radius: Double,
        seedMedian: Double,
        appearanceTolerance: Double
    ): Double {

        val sampleRadius =
            radius * 1.30

        var totalSupport = 0.0
        var sampleCount = 0

        for (angleIndex in 0 until ANGLE_SAMPLES) {

            val angle =
                2.0 * Math.PI *
                        angleIndex /
                        ANGLE_SAMPLES

            val x =
                centre.x +
                        sampleRadius *
                        cos(angle)

            val y =
                centre.y +
                        sampleRadius *
                        sin(angle)

            val value =
                sampleGray(
                    gray,
                    x,
                    y
                )
                    ?: continue

            val difference =
                abs(
                    value -
                            seedMedian
                )

            val similarity =
                (
                        1.0 -
                                difference /
                                (
                                        appearanceTolerance +
                                                30.0
                                        )
                        )
                    .coerceIn(
                        0.0,
                        1.0
                    )

            totalSupport +=
                similarity

            sampleCount++
        }

        return if (sampleCount > 0) {
            totalSupport /
                    sampleCount
        } else {
            0.0
        }
    }

    private fun sampleGray(
        gray: Mat,
        x: Double,
        y: Double
    ): Double? {

        val ix = x.roundToInt()
        val iy = y.roundToInt()

        if (
            ix < 0 ||
            iy < 0 ||
            ix >= gray.cols() ||
            iy >= gray.rows()
        ) {
            return null
        }

        return gray.get(iy, ix)?.firstOrNull()
    }

    private fun sampleGradient(
        gradient: Mat,
        x: Double,
        y: Double
    ): Double? {

        val ix = x.roundToInt()
        val iy = y.roundToInt()

        if (
            ix < 0 ||
            iy < 0 ||
            ix >= gradient.cols() ||
            iy >= gradient.rows()
        ) {
            return null
        }

        return gradient.get(iy, ix)?.firstOrNull()
    }


    private fun makeRoi(
        click: Point,
        width: Int,
        height: Int
    ): Rect {

        val half =
            ROI_SIZE / 2

        val x =
            (
                    click.x.roundToInt() -
                            half
                    )
                .coerceIn(
                    0,
                    max(
                        0,
                        width - ROI_SIZE
                    )
                )

        val y =
            (
                    click.y.roundToInt() -
                            half
                    )
                .coerceIn(
                    0,
                    max(
                        0,
                        height - ROI_SIZE
                    )
                )

        val roiWidth =
            min(
                ROI_SIZE,
                width - x
            )

        val roiHeight =
            min(
                ROI_SIZE,
                height - y
            )

        return Rect(
            x,
            y,
            roiWidth,
            roiHeight
        )
    }

    private fun makeDetection(
        result: BoundaryResult,
        scale: Double,
        roiRect: Rect
    ): BlobDetection {

        val centreX =
            (
                    result.centre.x +
                            roiRect.x
                    ) / scale

        val centreY =
            (
                    result.centre.y +
                            roiRect.y
                    ) / scale

        val radius =
            result.radius /
                    scale

        val area =
            (
                    Math.PI *
                            radius *
                            radius
                    )
                .roundToInt()

        return BlobDetection(
            centre = Point2(
                centreX.toFloat(),
                centreY.toFloat()
            ),
            radiusPx =
                radius.toFloat(),
            areaPixels = area,
            solidity =
                result.coverage.toFloat()
        )
    }
}