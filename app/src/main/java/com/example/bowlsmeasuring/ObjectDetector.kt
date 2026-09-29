package com.example.bowlsmeasuring

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

object ObjectDetector {

    private const val MAX_WORKING_SIZE = 1024

    /*
     * Hough circle parameters.
     *
     * These are deliberately fairly broad because:
     *
     * - woods may be black or brown
     * - the visible boundary can be weak
     * - perspective makes the apparent circle imperfect
     */
    private const val MIN_RADIUS = 12.0
    private const val MAX_RADIUS = 110.0

    private const val MIN_DISTANCE_BETWEEN_CIRCLES = 18.0

    /*
     * Hough parameters.
     *
     * param1 = Canny high threshold
     * param2 = accumulator threshold
     *
     * Lower param2 gives more candidates but also more false positives.
     */
    private const val HOUGH_PARAM1 = 100.0
    private const val HOUGH_PARAM2 = 15.0

    private const val EDGE_SAMPLES = 96

    data class BlobDetection(
        val centre: Point2,
        val radiusPx: Float,
        val areaPixels: Int,
        val solidity: Float
    )

    private data class CircleCandidate(
        val centre: Point,
        val radius: Double
    )

    private data class ScoredCandidate(
        val candidate: CircleCandidate,
        val score: Double
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
                            max(source.cols(), source.rows()).toDouble()
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

                /*
                 * Use a reasonably large ROI around the click.
                 *
                 * This prevents things elsewhere in the photograph
                 * from generating Hough circles.
                 */
                val roiSize = 360

                val x1 =
                    max(
                        0,
                        (click.x - roiSize / 2).roundToInt()
                    )

                val y1 =
                    max(
                        0,
                        (click.y - roiSize / 2).roundToInt()
                    )

                val x2 =
                    min(
                        working.cols(),
                        x1 + roiSize
                    )

                val y2 =
                    min(
                        working.rows(),
                        y1 + roiSize
                    )

                val roiRect = Rect(
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                )

                if (roiRect.width <= 20 || roiRect.height <= 20) {
                    return null
                }

                val roi = Mat(working, roiRect)

                try {

                    val localClick = Point(
                        click.x - roiRect.x,
                        click.y - roiRect.y
                    )

                    val circles =
                        findHoughCircles(roi)

                    if (circles.isEmpty()) {
                        return null
                    }

                    val scored =
                        mutableListOf<ScoredCandidate>()

                    for (candidate in circles) {

                        /*
                         * IMPORTANT:
                         *
                         * The clicked point must actually be inside
                         * the candidate circle.
                         *
                         * This stops a nearby shadow/grass circle
                         * becoming the answer when the user clicked
                         * directly on the wood.
                         */
                        val clickDistance =
                            distance(
                                candidate.centre,
                                localClick
                            )

                        /*
                         * The user is expected to click roughly at the centre
                         * of the object. Reject circles whose centre is too far
                         * from the click.
                         *
                         * This prevents a large circle whose circumference passes
                         * through the click from being accepted.
                         */
                        if (clickDistance > candidate.radius * 0.35) {
                            continue
                        }

                        /*
                         * Score the actual circle against the image.
                         */
                        val score =
                            scoreCircle(
                                roi,
                                candidate
                            )

                        if (score.isFinite()) {
                            scored +=
                                ScoredCandidate(
                                    candidate,
                                    score
                                )
                        }
                    }

                    if (scored.isEmpty()) {
                        return null
                    }

                    /*
                     * Sort strongest first.
                     */
                    scored.sortByDescending { it.score }

                    /*
                     * Prefer the smallest credible circle when
                     * several circles have nearly identical scores.
                     *
                     * This is important for the wood+shadow problem:
                     * a large circle can often get good edge coverage
                     * simply because it includes lots of image structure.
                     */
                    /*
                     * At this stage, proximity to the click is more important
                     * than finding the mathematically strongest edge.
                     *
                     * The click is our strongest piece of information about
                     * which object the user wants.
                     */
                    val chosen =
                        scored.minWithOrNull(
                            compareBy<ScoredCandidate> {
                                distance(
                                    it.candidate.centre,
                                    localClick
                                )
                            }.thenByDescending {
                                it.score
                            }
                        ) ?: return null

                    return makeDetection(
                        candidate = chosen.candidate,
                        scale = scale,
                        roiRect = roiRect
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

    fun createDebugImage(
        bitmap: Bitmap,
        clickedPoint: Point2
    ): Bitmap? {

        val source = Mat()

        try {
            Utils.bitmapToMat(bitmap, source)

            val scale =
                min(
                    1.0,
                    MAX_WORKING_SIZE.toDouble() /
                            max(source.cols(), source.rows()).toDouble()
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

                val roiSize = 360

                val x1 =
                    max(
                        0,
                        (click.x - roiSize / 2).roundToInt()
                    )

                val y1 =
                    max(
                        0,
                        (click.y - roiSize / 2).roundToInt()
                    )

                val x2 =
                    min(
                        working.cols(),
                        x1 + roiSize
                    )

                val y2 =
                    min(
                        working.rows(),
                        y1 + roiSize
                    )

                val roiRect = Rect(
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                )

                val roi = Mat(working, roiRect)

                try {
                    val localClick = Point(
                        click.x - roiRect.x,
                        click.y - roiRect.y
                    )

                    /*
                     * Draw over a colour copy of the ROI.
                     */
                    val debug = Mat()
                    roi.copyTo(debug)

                    val circles =
                        findHoughCircles(roi)

                    /*
                     * Work out which circles pass our current
                     * centre-distance constraint.
                     */
                    val scored =
                        mutableListOf<ScoredCandidate>()

                    for (candidate in circles) {

                        val clickDistance =
                            distance(
                                candidate.centre,
                                localClick
                            )

                        if (
                            clickDistance <=
                            candidate.radius * 0.35
                        ) {

                            val score =
                                scoreCircle(
                                    roi,
                                    candidate
                                )

                            if (score.isFinite()) {
                                scored +=
                                    ScoredCandidate(
                                        candidate,
                                        score
                                    )
                            }
                        }
                    }

                    /*
                     * Find the candidate our current detector
                     * would actually select.
                     */
                    val selected =
                        scored.minWithOrNull(
                            compareBy<ScoredCandidate> {
                                distance(
                                    it.candidate.centre,
                                    localClick
                                )
                            }.thenByDescending {
                                it.score
                            }
                        )?.candidate

                    /*
                     * Draw every Hough candidate.
                     *
                     * RED:
                     *     rejected by centre-distance constraint
                     *
                     * GREEN:
                     *     passed centre-distance constraint
                     *
                     * BLUE:
                     *     selected candidate
                     */
                    for (candidate in circles) {

                        val clickDistance =
                            distance(
                                candidate.centre,
                                localClick
                            )

                        val passes =
                            clickDistance <=
                                    candidate.radius * 0.35

                        val isSelected =
                            selected != null &&
                                    distance(
                                        candidate.centre,
                                        selected.centre
                                    ) < 0.01 &&
                                    abs(
                                        candidate.radius -
                                                selected.radius
                                    ) < 0.01

                        val colour =
                            when {
                                isSelected ->
                                    Scalar(0.0, 0.0, 255.0, 64.0) // blue

                                passes ->
                                    Scalar(0.0, 255.0, 0.0, 64.0) // green

                                else ->
                                    Scalar(255.0, 0.0, 0.0, 64.0) // red
                            }

                        Imgproc.circle(
                            debug,
                            candidate.centre,
                            candidate.radius.roundToInt(),
                            colour,
                            2
                        )

                        /*
                         * Draw the candidate centre.
                         */
                        Imgproc.circle(
                            debug,
                            candidate.centre,
                            3,
                            colour,
                            Imgproc.FILLED
                        )
                    }

                    /*
                     * Draw the click point as a yellow cross.
                     */
                    val crossSize = 10

                    Imgproc.line(
                        debug,
                        Point(
                            localClick.x - crossSize,
                            localClick.y
                        ),
                        Point(
                            localClick.x + crossSize,
                            localClick.y
                        ),
                        Scalar(0.0, 255.0, 255.0, 255.0),
                        2
                    )

                    Imgproc.line(
                        debug,
                        Point(
                            localClick.x,
                            localClick.y - crossSize
                        ),
                        Point(
                            localClick.x,
                            localClick.y + crossSize
                        ),
                        Scalar(0.0, 255.0, 255.0, 255.0),
                        2
                    )

                    /*
                     * Convert back to Bitmap.
                     */
                    val output =
                        Bitmap.createBitmap(
                            debug.cols(),
                            debug.rows(),
                            Bitmap.Config.ARGB_8888
                        )

                    Utils.matToBitmap(
                        debug,
                        output
                    )

                    debug.release()

                    return output

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

    private fun findHoughCircles(
        roi: Mat
    ): List<CircleCandidate> {

        val gray = Mat()

        try {

            Imgproc.cvtColor(
                roi,
                gray,
                Imgproc.COLOR_RGBA2GRAY
            )

            /*
             * Mild blur suppresses grass texture while retaining
             * the relatively strong object boundary.
             */
            Imgproc.GaussianBlur(
                gray,
                gray,
                Size(5.0, 5.0),
                1.2
            )

            val circles = Mat()

            try {

                Imgproc.HoughCircles(
                    gray,
                    circles,
                    Imgproc.HOUGH_GRADIENT,
                    1.0,
                    MIN_DISTANCE_BETWEEN_CIRCLES,
                    HOUGH_PARAM1,
                    HOUGH_PARAM2,
                    MIN_RADIUS.roundToInt(),
                    MAX_RADIUS.roundToInt()
                )

                val result =
                    mutableListOf<CircleCandidate>()

                for (i in 0 until circles.cols()) {

                    val data =
                        circles.get(0, i)
                            ?: continue

                    if (data.size < 3) {
                        continue
                    }

                    result +=
                        CircleCandidate(
                            centre =
                                Point(
                                    data[0],
                                    data[1]
                                ),
                            radius = data[2]
                        )
                }

                return result

            } finally {
                circles.release()
            }

        } finally {
            gray.release()
        }
    }

    private fun scoreCircle(
        image: Mat,
        candidate: CircleCandidate
    ): Double {

        val gray = Mat()

        try {

            Imgproc.cvtColor(
                image,
                gray,
                Imgproc.COLOR_RGBA2GRAY
            )

            /*
             * Calculate a gradient image.
             */
            val gx = Mat()
            val gy = Mat()

            try {

                Imgproc.Sobel(
                    gray,
                    gx,
                    CvType.CV_32F,
                    1,
                    0,
                    3
                )

                Imgproc.Sobel(
                    gray,
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

                    var edgeSum = 0.0
                    var edgeSquaredSum = 0.0
                    var validSamples = 0

                    val radialSamples = 3

                    /*
                     * Sample slightly inside, on, and outside
                     * the proposed circle.
                     */
                    for (i in 0 until EDGE_SAMPLES) {

                        val angle =
                            2.0 * Math.PI *
                                    i / EDGE_SAMPLES

                        val cosA = cos(angle)
                        val sinA = sin(angle)

                        var bestGradient = 0.0

                        for (rIndex in 0 until radialSamples) {

                            val radialOffset =
                                (rIndex - 1) * 2.0

                            val x =
                                candidate.centre.x +
                                        (candidate.radius + radialOffset) *
                                        cosA

                            val y =
                                candidate.centre.y +
                                        (candidate.radius + radialOffset) *
                                        sinA

                            val ix =
                                x.roundToInt()

                            val iy =
                                y.roundToInt()

                            if (
                                ix < 0 ||
                                iy < 0 ||
                                ix >= magnitude.cols() ||
                                iy >= magnitude.rows()
                            ) {
                                continue
                            }

                            val value =
                                magnitude.get(iy, ix)
                                    ?.firstOrNull()
                                    ?: continue

                            if (value > bestGradient) {
                                bestGradient = value
                            }
                        }

                        edgeSum += bestGradient
                        edgeSquaredSum +=
                            bestGradient * bestGradient

                        validSamples++
                    }

                    if (validSamples == 0) {
                        return Double.NEGATIVE_INFINITY
                    }

                    val meanEdge =
                        edgeSum / validSamples

                    val variance =
                        (
                                edgeSquaredSum /
                                        validSamples
                                ) -
                                meanEdge * meanEdge

                    val standardDeviation =
                        sqrt(
                            max(
                                0.0,
                                variance
                            )
                        )

                    /*
                     * A good object boundary should have:
                     *
                     * - reasonably strong gradients
                     * - reasonably consistent gradients
                     *
                     * Consistency prevents one strong grass/shadow
                     * edge from dominating the result.
                     */
                    val consistency =
                        if (meanEdge > 1e-6) {
                            meanEdge /
                                    (meanEdge + standardDeviation)
                        } else {
                            0.0
                        }

                    /*
                     * Compare brightness just inside and outside
                     * the circle.
                     *
                     * This is useful for both brown and black woods.
                     * We don't assume the object is dark — we only
                     * look for a boundary between the two regions.
                     */
                    var insideSum = 0.0
                    var outsideSum = 0.0
                    var contrastSamples = 0

                    for (i in 0 until EDGE_SAMPLES) {

                        val angle =
                            2.0 * Math.PI *
                                    i / EDGE_SAMPLES

                        val cosA = cos(angle)
                        val sinA = sin(angle)

                        val insideX =
                            candidate.centre.x +
                                    candidate.radius * 0.75 *
                                    cosA

                        val insideY =
                            candidate.centre.y +
                                    candidate.radius * 0.75 *
                                    sinA

                        val outsideX =
                            candidate.centre.x +
                                    candidate.radius * 1.20 *
                                    cosA

                        val outsideY =
                            candidate.centre.y +
                                    candidate.radius * 1.20 *
                                    sinA

                        val inside =
                            sampleGray(
                                gray,
                                insideX,
                                insideY
                            )

                        val outside =
                            sampleGray(
                                gray,
                                outsideX,
                                outsideY
                            )

                        if (
                            inside != null &&
                            outside != null
                        ) {
                            insideSum += inside
                            outsideSum += outside
                            contrastSamples++
                        }
                    }

                    val contrast =
                        if (contrastSamples > 0) {
                            abs(
                                insideSum /
                                        contrastSamples -
                                        outsideSum /
                                        contrastSamples
                            )
                        } else {
                            0.0
                        }

                    /*
                     * Normalise the two useful signals.
                     *
                     * Sobel values can be considerably larger than
                     * 255, so don't use a fixed threshold here.
                     */
                    val edgeScore =
                        meanEdge /
                                (meanEdge + 20.0)

                    val contrastScore =
                        contrast /
                                (contrast + 20.0)

                    return (
                            edgeScore * 0.55 +
                                    consistency * 0.20 +
                                    contrastScore * 0.25
                            )

                } finally {
                    magnitude.release()
                }

            } finally {
                gx.release()
                gy.release()
            }

        } finally {
            gray.release()
        }
    }

    private fun sampleGray(
        image: Mat,
        x: Double,
        y: Double
    ): Double? {

        val ix = x.roundToInt()
        val iy = y.roundToInt()

        if (
            ix < 0 ||
            iy < 0 ||
            ix >= image.cols() ||
            iy >= image.rows()
        ) {
            return null
        }

        return image.get(iy, ix)
            ?.firstOrNull()
    }

    private fun makeDetection(
        candidate: CircleCandidate,
        scale: Double,
        roiRect: Rect
    ): BlobDetection {

        /*
         * Convert from working-image/ROI coordinates back into
         * original bitmap coordinates.
         */
        val centreX =
            (
                    candidate.centre.x +
                            roiRect.x
                    ) / scale

        val centreY =
            (
                    candidate.centre.y +
                            roiRect.y
                    ) / scale

        val radius =
            candidate.radius /
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
            radiusPx = radius.toFloat(),
            areaPixels = area,
            solidity = 1.0f
        )
    }

    private fun distance(
        a: Point,
        b: Point
    ): Double {

        val dx = a.x - b.x
        val dy = a.y - b.y

        return sqrt(
            dx * dx +
                    dy * dy
        )
    }
}