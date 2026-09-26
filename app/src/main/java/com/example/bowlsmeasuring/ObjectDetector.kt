package com.example.bowlsmeasuring

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.geometry.Geometry
import org.opencv.imgproc.Imgproc
import kotlin.math.*

object ObjectDetector {

    private const val MAX_WORKING_SIZE = 1024

    /*
     * The blur is deliberately heavy. We are looking for objects at roughly
     * the scale of bowls/woods, not individual blades of grass.
     */
    private val BLUR_SIGMAS = doubleArrayOf(
        6.0,
        8.0,
        11.0,
        15.0,
        20.0,
        26.0
    )

    private val BLUR_KERNELS = intArrayOf(
        25,
        31,
        41,
        51,
        71,
        91
    )

    /*
     * This is ONLY an image-processing relationship between blur scale and
     * approximate object radius. It has nothing to do with the physical
     * jack/wood diameter ratio.
     */
    private const val SCALE_TO_RADIUS = 1.5

    private const val ROI_SIZE = 360

    /*
     * Number of angular samples used when evaluating a candidate ring.
     */
    private const val RING_SAMPLES = 96

    /*
     * Fraction of the candidate circumference which must have reasonably
     * strong boundary evidence.
     */
    private const val MIN_RING_COVERAGE = 0.42

    /*
     * Maximum permitted eccentricity during the circular search.
     *
     * We deliberately allow quite a lot because perspective can make a
     * circular wood look elliptical.
     */
    private const val MAX_ECCENTRICITY = 0.45

    data class BlobDetection(
        val centre: Point2,
        val radiusPx: Float,
        val areaPixels: Int,
        val solidity: Float
    )

    private data class CandidateCentre(
        val centre: Point,
        val estimatedRadius: Double,
        val response: Double
    )

    private data class RingResult(
        val centre: Point,
        val radius: Double,
        val score: Double,
        val coverage: Double,
        val meanEdge: Double,
        val consistency: Double
    )

    private data class RefinedCandidate(
        val detection: BlobDetection,
        val score: Double,
        val clickDistance: Double,
        val ring: RingResult
    )

    /**
     * Detect the physical object nearest to the user's click.
     *
     * The click defines the area in which we expect the object to be.
     * It does NOT itself become an artificial detection.
     */
    fun detectNear(
        bitmap: Bitmap,
        clickedPoint: Point2
    ): BlobDetection? {

        val source = Mat()

        try {
            Utils.bitmapToMat(bitmap, source)

            val scale = min(
                1.0,
                MAX_WORKING_SIZE.toDouble() /
                        max(source.cols(), source.rows()).toDouble()
            )

            val working = Mat()

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

            val roiRect = makeRoi(
                click,
                working.cols(),
                working.rows()
            )

            val roi = Mat(working, roiRect)

            val localClick = Point(
                click.x - roiRect.x,
                click.y - roiRect.y
            )

            val candidates = findCandidateCentres(
                roi
            )

            if (candidates.isEmpty()) {
                return null
            }

            val refined = mutableListOf<RefinedCandidate>()

            for (candidate in candidates) {

                val candidateDistance = distance(
                    candidate.centre,
                    localClick
                )

                /*
                 * Don't allow a candidate on the opposite side of the ROI
                 * to win merely because it has a strong edge.
                 */
                if (candidateDistance > ROI_SIZE * 0.55) {
                    continue
                }

                val ring = findBestRing(
                    roi,
                    candidate.centre,
                    candidate.estimatedRadius
                ) ?: continue

                /*
                 * The user's click MUST be inside the detected circle.
                 *
                 * This is a hard constraint, not a scoring preference.
                 *
                 * The click does not need to be near the centre of the
                 * circle: the user may deliberately click off-centre on a
                 * wood, jack or shadow.
                 */
                val clickInsideDistance = distance(
                    ring.centre,
                    localClick
                )

                if (clickInsideDistance > ring.radius) {
                    continue
                }

                if (ring.coverage < MIN_RING_COVERAGE) {
                    continue
                }

                val detection = makeDetection(
                    roi = roi,
                    ring = ring,
                    scale = scale,
                    roiRect = roiRect
                ) ?: continue

                /*
                 * At this point the candidate has passed the hard click-containment
                 * test, so proximity is only a small tie-breaker.
                 */
                val proximity =
                    1.0 -
                            (candidateDistance / 100.0)
                                .coerceIn(0.0, 1.0)

                val finalScore =
                    ring.score +
                            proximity * 0.10

                refined += RefinedCandidate(
                    detection = detection,
                    score = finalScore,
                    clickDistance = clickInsideDistance,
                    ring = ring
                )
            }

            /*
             * Do NOT simply take the closest candidate.
             *
             * A shadow can be very close to the click but have poor closed
             * ring coherence.
             */
            return refined
                .maxByOrNull { it.score }
                ?.detection

        } finally {
            source.release()
        }
    }

    // ---------------------------------------------------------------------
    // Candidate detection
    // ---------------------------------------------------------------------

    private fun findCandidateCentres(
        roi: Mat
    ): List<CandidateCentre> {

        val gray = Mat()

        try {
            Imgproc.cvtColor(
                roi,
                gray,
                Imgproc.COLOR_RGBA2GRAY
            )

            val candidates = mutableListOf<CandidateCentre>()

            for (i in BLUR_SIGMAS.indices) {

                val sigma = BLUR_SIGMAS[i]
                val kernel = BLUR_KERNELS[i]

                val blurred = Mat()

                try {
                    Imgproc.GaussianBlur(
                        gray,
                        blurred,
                        Size(kernel.toDouble(), kernel.toDouble()),
                        sigma,
                        sigma,
                        Core.BORDER_REPLICATE
                    )

                    /*
                     * Large-scale local contrast.
                     *
                     * We compare the heavily blurred image with an even
                     * more heavily blurred background. This suppresses grass
                     * texture and leaves larger structures.
                     */
                    val background = Mat()

                    try {
                        val bgKernel = kernel * 2 + 1
                        val bgSigma = sigma * 2.0

                        Imgproc.GaussianBlur(
                            blurred,
                            background,
                            Size(
                                bgKernel.toDouble(),
                                bgKernel.toDouble()
                            ),
                            bgSigma,
                            bgSigma,
                            Core.BORDER_REPLICATE
                        )

                        val difference = Mat()

                        try {
                            Core.absdiff(
                                blurred,
                                background,
                                difference
                            )

                            val normalised = Mat()

                            try {
                                Core.normalize(
                                    difference,
                                    normalised,
                                    0.0,
                                    255.0,
                                    Core.NORM_MINMAX,
                                    CvType.CV_16UC1
                                )

                                /*
                                 * Threshold only the strongest large-scale
                                 * responses.
                                 */
                                val threshold = Mat()

                                try {
                                    Imgproc.threshold(
                                        normalised,
                                        threshold,
                                        0.0,
                                        255.0,
                                        Imgproc.THRESH_BINARY or
                                                Imgproc.THRESH_OTSU
                                    )

                                    threshold.convertTo(
                                        threshold,
                                        CvType.CV_8U
                                    )

                                    /*
                                     * Remove tiny components. The opening
                                     * also helps suppress residual grass.
                                     */
                                    val morphKernel =
                                        Imgproc.getStructuringElement(
                                            Imgproc.MORPH_ELLIPSE,
                                            Size(9.0, 9.0)
                                        )

                                    Imgproc.morphologyEx(
                                        threshold,
                                        threshold,
                                        Imgproc.MORPH_OPEN,
                                        morphKernel
                                    )

                                    val contours =
                                        mutableListOf<MatOfPoint>()

                                    Imgproc.findContours(
                                        threshold,
                                        contours,
                                        Mat(),
                                        Imgproc.RETR_EXTERNAL,
                                        Imgproc.CHAIN_APPROX_SIMPLE
                                    )

                                    for (contour in contours) {

                                        val area =
                                            Geometry.contourArea(contour)

                                        if (area < 150.0) {
                                            contour.release()
                                            continue
                                        }

                                        val moments =
                                            Geometry.moments(contour)

                                        if (abs(moments.m00) < 1e-6) {
                                            contour.release()
                                            continue
                                        }

                                        val cx =
                                            moments.m10 / moments.m00

                                        val cy =
                                            moments.m01 / moments.m00

                                        val radius =
                                            sigma *
                                                    SCALE_TO_RADIUS

                                        /*
                                         * Candidate response is simply the
                                         * average large-scale response in
                                         * the component.
                                         */
                                        val response =
                                            Geometry.contourArea(contour) /
                                                    (roi.cols() *
                                                            roi.rows()).toDouble()

                                        candidates += CandidateCentre(
                                            centre = Point(cx, cy),
                                            estimatedRadius = radius,
                                            response = response
                                        )

                                        contour.release()
                                    }

                                } finally {
                                    threshold.release()
                                }

                            } finally {
                                normalised.release()
                            }

                        } finally {
                            difference.release()
                        }

                    } finally {
                        background.release()
                    }

                } finally {
                    blurred.release()
                }
            }

            /*
             * Merge candidates that are effectively the same location.
             */
            return mergeCandidates(candidates)

        } finally {
            gray.release()
        }
    }

    private fun mergeCandidates(
        candidates: List<CandidateCentre>
    ): List<CandidateCentre> {

        val result = mutableListOf<CandidateCentre>()

        for (candidate in candidates) {

            val existingIndex =
                result.indexOfFirst {
                    distance(it.centre, candidate.centre) < 25.0
                }

            if (existingIndex < 0) {
                result += candidate
            } else {
                val existing = result[existingIndex]

                /*
                 * Keep the candidate with the stronger response.
                 */
                if (candidate.response > existing.response) {
                    result[existingIndex] = candidate
                }
            }
        }

        return result
    }

    // ---------------------------------------------------------------------
    // Ring search
    // ---------------------------------------------------------------------

    private fun findBestRing(
        image: Mat,
        initialCentre: Point,
        estimatedRadius: Double
    ): RingResult? {

        val gray = Mat()

        try {
            Imgproc.cvtColor(
                image,
                gray,
                Imgproc.COLOR_RGBA2GRAY
            )

            /*
             * Light blur for stable edge measurements.
             *
             * We do NOT use the very heavily blurred image here because we
             * want the actual physical boundary.
             */
            val smooth = Mat()

            try {
                Imgproc.GaussianBlur(
                    gray,
                    smooth,
                    Size(5.0, 5.0),
                    1.5,
                    1.5,
                    Core.BORDER_REPLICATE
                )

                /*
                 * Gradient magnitude.
                 */
                val gx = Mat()
                val gy = Mat()

                try {
                    Imgproc.Sobel(
                        smooth,
                        gx,
                        CvType.CV_32F,
                        1,
                        0,
                        3
                    )

                    Imgproc.Sobel(
                        smooth,
                        gy,
                        CvType.CV_32F,
                        0,
                        1,
                        3
                    )

                    val magnitude = Mat()

                    try {
                        Core.magnitude(
                            gx,
                            gy,
                            magnitude
                        )

                        /*
                         * Search a reasonably broad radius range.
                         *
                         * This is important: we are no longer trusting the
                         * blur-derived radius.
                         */
                        val minRadius =
                            max(
                                10.0,
                                estimatedRadius * 0.55
                            )

                        val maxRadius =
                            min(
                                min(
                                    image.cols(),
                                    image.rows()
                                ) * 0.40,
                                estimatedRadius * 1.80
                            )

                        if (maxRadius <= minRadius) {
                            return null
                        }

                        /*
                         * Also search small centre offsets. The candidate
                         * centre is only approximate.
                         */
                        var best: RingResult? = null

                        val centreStep =
                            max(
                                3.0,
                                estimatedRadius * 0.12
                            )

                        val centreOffsets = listOf(
                            Point(0.0, 0.0),
                            Point(-centreStep, 0.0),
                            Point(centreStep, 0.0),
                            Point(0.0, -centreStep),
                            Point(0.0, centreStep),
                            Point(-centreStep, -centreStep),
                            Point(centreStep, -centreStep),
                            Point(-centreStep, centreStep),
                            Point(centreStep, centreStep)
                        )

                        var radius =
                            minRadius

                        while (radius <= maxRadius) {

                            for (offset in centreOffsets) {

                                val centre = Point(
                                    initialCentre.x + offset.x,
                                    initialCentre.y + offset.y
                                )

                                val result =
                                    scoreRing(
                                        magnitude,
                                        centre,
                                        radius
                                    )

                                if (result != null) {

                                    val candidate =
                                        result.copy(
                                            centre = centre,
                                            radius = radius
                                        )

                                    if (best == null ||
                                        candidate.score >
                                        best!!.score
                                    ) {
                                        best = candidate
                                    }
                                }
                            }

                            /*
                             * Fine enough for the first pass without making
                             * the detector unnecessarily expensive.
                             */
                            radius += max(
                                2.0,
                                estimatedRadius * 0.04
                            )
                        }

                        return best

                    } finally {
                        magnitude.release()
                    }

                } finally {
                    gx.release()
                    gy.release()
                }

            } finally {
                smooth.release()
            }

        } finally {
            gray.release()
        }
    }

    /**
     * Score a complete ring.
     *
     * The key idea:
     *
     * A real wood should produce boundary evidence at a large fraction of
     * the circumference.
     *
     * A shadow edge should generally produce evidence over only part of it.
     */
    private fun scoreRing(
        gradient: Mat,
        centre: Point,
        radius: Double
    ): RingResult? {

        /*
         * Reject rings which don't fit inside the image.
         */
        if (centre.x - radius - 4 < 0 ||
            centre.y - radius - 4 < 0 ||
            centre.x + radius + 4 >= gradient.cols() ||
            centre.y + radius + 4 >= gradient.rows()
        ) {
            return null
        }

        val values = DoubleArray(RING_SAMPLES)

        var valid = 0

        for (i in 0 until RING_SAMPLES) {

            val theta =
                2.0 * Math.PI *
                        i.toDouble() /
                        RING_SAMPLES.toDouble()

            val x =
                centre.x +
                        cos(theta) * radius

            val y =
                centre.y +
                        sin(theta) * radius

            val value =
                sampleGradient(
                    gradient,
                    x,
                    y
                )

            if (value.isFinite()) {
                values[i] = value
                valid++
            }
        }

        if (valid < RING_SAMPLES * 0.90) {
            return null
        }

        /*
         * Robust local threshold.
         *
         * We don't use one global gradient threshold because lighting,
         * grass colour and image exposure vary substantially.
         */
        val sorted =
            values.sorted()

        val median =
            sorted[sorted.size / 2]

        val upperQuartile =
            sorted[
                (sorted.size * 0.75)
                    .toInt()
                    .coerceAtMost(sorted.lastIndex)
            ]

        val noiseFloor =
            max(
                1.0,
                median * 1.25
            )

        val threshold =
            max(
                noiseFloor,
                upperQuartile * 0.55
            )

        var strongCount = 0
        var totalStrength = 0.0

        for (value in values) {

            if (value >= threshold) {
                strongCount++
            }

            totalStrength += value
        }

        val coverage =
            strongCount.toDouble() /
                    values.size.toDouble()

        /*
         * Mean strength of the complete ring.
         */
        val meanEdge =
            totalStrength /
                    values.size.toDouble()

        /*
         * Measure angular consistency.
         *
         * A genuine boundary should not have all its energy concentrated in
         * one small section.
         */
        val mean =
            meanEdge

        var variance = 0.0

        for (value in values) {
            val d = value - mean
            variance += d * d
        }

        variance /=
            values.size.toDouble()

        val standardDeviation =
            sqrt(variance)

        val consistency =
            if (mean <= 1e-6) {
                0.0
            } else {
                (
                        1.0 -
                                standardDeviation /
                                (mean * 2.0)
                        )
                    .coerceIn(0.0, 1.0)
            }

        /*
         * Coverage is the most important term.
         *
         * Mean edge strength alone would favour shadows because shadow
         * boundaries can actually be stronger than wood boundaries.
         */
        val coverageScore =
            coverage.pow(1.7)

        val strengthScore =
            (meanEdge / 20.0)
                .coerceIn(0.0, 1.0)

        val score =
            coverageScore * 0.60 +
                    strengthScore * 0.20 +
                    consistency * 0.20

        return RingResult(
            centre = centre,
            radius = radius,
            score = score,
            coverage = coverage,
            meanEdge = meanEdge,
            consistency = consistency
        )
    }

    // ---------------------------------------------------------------------
    // Detection / ellipse refinement
    // ---------------------------------------------------------------------

    private fun makeDetection(
        roi: Mat,
        ring: RingResult,
        scale: Double,
        roiRect: Rect
    ): BlobDetection? {

        val gray = Mat()

        try {
            Imgproc.cvtColor(
                roi,
                gray,
                Imgproc.COLOR_RGBA2GRAY
            )

            val edges = Mat()

            try {

                Imgproc.Canny(
                    gray,
                    edges,
                    40.0,
                    120.0
                )

                /*
                 * Collect edge points close to the selected ring.
                 */
                val points = mutableListOf<Point>()

                val searchBand =
                    max(
                        5.0,
                        ring.radius * 0.12
                    )

                val minX =
                    max(
                        0,
                        floor(
                            ring.centre.x -
                                    ring.radius -
                                    searchBand
                        ).toInt()
                    )

                val maxX =
                    min(
                        roi.cols() - 1,
                        ceil(
                            ring.centre.x +
                                    ring.radius +
                                    searchBand
                        ).toInt()
                    )

                val minY =
                    max(
                        0,
                        floor(
                            ring.centre.y -
                                    ring.radius -
                                    searchBand
                        ).toInt()
                    )

                val maxY =
                    min(
                        roi.rows() - 1,
                        ceil(
                            ring.centre.y +
                                    ring.radius +
                                    searchBand
                        ).toInt()
                    )

                for (y in minY..maxY) {

                    for (x in minX..maxX) {

                        if (edges.get(y, x)[0] < 1.0) {
                            continue
                        }

                        val dx =
                            x -
                                    ring.centre.x

                        val dy =
                            y -
                                    ring.centre.y

                        val distance =
                            sqrt(
                                dx * dx +
                                        dy * dy
                            )

                        if (
                            abs(distance - ring.radius)
                            <= searchBand
                        ) {
                            points += Point(
                                x.toDouble(),
                                y.toDouble()
                            )
                        }
                    }
                }

                /*
                 * We need enough points for a meaningful ellipse.
                 */
                if (points.size < 20) {
                    return simpleCircularDetection(
                        ring,
                        scale,
                        roiRect
                    )
                }

                val contour =
                    MatOfPoint2f()

                try {
                    contour.fromList(points)

                    val ellipse =
                        try {
                            Geometry.fitEllipse(contour)
                        } catch (_: Exception) {
                            null
                        }

                    if (ellipse == null) {
                        return simpleCircularDetection(
                            ring,
                            scale,
                            roiRect
                        )
                    }

                    /*
                     * fitEllipse gives full axis lengths.
                     */
                    val major =
                        max(
                            ellipse.size.width,
                            ellipse.size.height
                        )

                    val minor =
                        min(
                            ellipse.size.width,
                            ellipse.size.height
                        )

                    if (minor <= 1.0) {
                        return simpleCircularDetection(
                            ring,
                            scale,
                            roiRect
                        )
                    }

                    val eccentricity =
                        1.0 -
                                minor / major

                    /*
                     * If the ellipse is wildly elongated it is much more
                     * likely to be a shadow or unrelated edge.
                     */
                    if (eccentricity > MAX_ECCENTRICITY) {
                        return simpleCircularDetection(
                            ring,
                            scale,
                            roiRect
                        )
                    }

                    val centreX =
                        (
                                ellipse.center.x +
                                        roiRect.x
                                ) / scale

                    val centreY =
                        (
                                ellipse.center.y +
                                        roiRect.y
                                ) / scale

                    val radiusPx =
                        (
                                (major + minor) / 4.0
                                ) / scale

                    val area =
                        (
                                Math.PI *
                                        (major / 2.0) *
                                        (minor / 2.0)
                                ).toInt()

                    return BlobDetection(
                        centre = Point2(
                            centreX.toFloat(),
                            centreY.toFloat()
                        ),
                        radiusPx = radiusPx.toFloat(),
                        areaPixels = area,
                        solidity = ring.coverage.toFloat()
                    )

                } finally {
                    contour.release()
                }

            } finally {
                edges.release()
            }

        } finally {
            gray.release()
        }
    }

    private fun simpleCircularDetection(
        ring: RingResult,
        scale: Double,
        roiRect: Rect
    ): BlobDetection {

        val centreX =
            (
                    ring.centre.x +
                            roiRect.x
                    ) / scale

        val centreY =
            (
                    ring.centre.y +
                            roiRect.y
                    ) / scale

        val radius =
            ring.radius / scale

        return BlobDetection(
            centre = Point2(
                centreX.toFloat(),
                centreY.toFloat()
            ),
            radiusPx = radius.toFloat(),
            areaPixels = (
                    Math.PI *
                            radius *
                            radius
                    ).toInt(),
            solidity = ring.coverage.toFloat()
        )
    }

    // ---------------------------------------------------------------------
    // Image / geometry helpers
    // ---------------------------------------------------------------------

    private fun makeRoi(
        click: Point,
        width: Int,
        height: Int
    ): Rect {

        val half =
            ROI_SIZE / 2

        val left =
            (click.x - half)
                .toInt()
                .coerceIn(
                    0,
                    max(0, width - ROI_SIZE)
                )

        val top =
            (click.y - half)
                .toInt()
                .coerceIn(
                    0,
                    max(0, height - ROI_SIZE)
                )

        val roiWidth =
            min(
                ROI_SIZE,
                width - left
            )

        val roiHeight =
            min(
                ROI_SIZE,
                height - top
            )

        return Rect(
            left,
            top,
            roiWidth,
            roiHeight
        )
    }

    private fun sampleGradient(
        image: Mat,
        x: Double,
        y: Double
    ): Double {

        val x0 =
            floor(x).toInt()

        val y0 =
            floor(y).toInt()

        val x1 =
            min(
                image.cols() - 1,
                x0 + 1
            )

        val y1 =
            min(
                image.rows() - 1,
                y0 + 1
            )

        if (
            x0 < 0 ||
            y0 < 0 ||
            x0 >= image.cols() ||
            y0 >= image.rows()
        ) {
            return Double.NaN
        }

        val fx =
            x - x0

        val fy =
            y - y0

        val v00 =
            image.get(y0, x0)[0]

        val v10 =
            image.get(y0, x1)[0]

        val v01 =
            image.get(y1, x0)[0]

        val v11 =
            image.get(y1, x1)[0]

        val top =
            v00 +
                    (v10 - v00) * fx

        val bottom =
            v01 +
                    (v11 - v01) * fx

        return top +
                (bottom - top) * fy
    }

    private fun distance(
        a: Point,
        b: Point
    ): Double {

        val dx =
            a.x - b.x

        val dy =
            a.y - b.y

        return sqrt(
            dx * dx +
                    dy * dy
        )
    }
}