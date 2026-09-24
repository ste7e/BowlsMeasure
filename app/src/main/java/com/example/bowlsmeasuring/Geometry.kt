package com.example.bowlsmeasuring

import kotlin.math.sqrt

data class Point2(val x: Float, val y: Float)
data class ObjectMeasurement(val name: String, val centre: Point2, val apparentRadiusPx: Float)
data class CorrectedMeasurement(val name: String, val relativeDistance: Double, val rank: Int)
data class PlaneModelResult(
    val measurements: List<CorrectedMeasurement>,
    val planeRms: Double,
    val focalLengthPx: Double
)

public data class Vec3(val x: Double, val y: Double, val z: Double) {
    operator fun plus(other: Vec3) = Vec3(x + other.x, y + other.y, z + other.z)
    operator fun minus(other: Vec3) = Vec3(x - other.x, y - other.y, z - other.z)
    operator fun times(s: Double) = Vec3(x * s, y * s, z * s)
    fun dot(other: Vec3): Double = x * other.x + y * other.y + z * other.z
    fun cross(other: Vec3) = Vec3(
        y * other.z - z * other.y,
        z * other.x - x * other.z,
        x * other.y - y * other.x
    )

    fun length(): Double = sqrt(dot(this))
    fun normalized(): Vec3 {
        val l = length()
        return if (l > 1e-12) this * (1.0 / l) else Vec3(0.0, 0.0, 1.0)
    }
}

private data class CameraPoint(val name: String, val point: Vec3)

/**
 * Experimental perspective model.
 *
 * Every bowl is assumed to have the same physical radius.  Its measured image
 * radius therefore gives a relative camera depth: Z ∝ 1 / apparentRadius.
 * The jack is converted to an equivalent wood radius using the known diameter
 * ratio.
 *
 * The resulting 3-D points are fitted to a single ground plane. Distances are
 * then measured in that plane rather than directly through 3-D space. This is
 * still an experiment, not a calibrated photogrammetry solution.
 */
object SizeDepthModel {
    fun calculate(
        jack: ObjectMeasurement,
        jackToWoodDiameterRatio: Double,
        focalLengthPx: Double,
        principalX: Double,
        principalY: Double,
        woods: List<ObjectMeasurement>
    ): PlaneModelResult? {
        if (woods.isEmpty() || jackToWoodDiameterRatio <= 0.0 || focalLengthPx <= 0.0) return null

        val all = buildList {
            add(jack)
            addAll(woods)
        }

        val points = all.map { measurement ->
            val radius = if (measurement.name == "Jack") {
                measurement.apparentRadiusPx.toDouble() / jackToWoodDiameterRatio
            } else {
                measurement.apparentRadiusPx.toDouble()
            }

            if (radius <= 0.0) return null

            // Coordinates are expressed in arbitrary units of wood radius.
            // X and Y simplify to image offset / apparent radius; Z retains f.
            val x = (measurement.centre.x.toDouble() - principalX) / radius
            val y = (measurement.centre.y.toDouble() - principalY) / radius
            val z = focalLengthPx / radius
            CameraPoint(measurement.name, Vec3(x, y, z))
        }

        val plane = fitPlane(points.map { it.point }) ?: return null
        val planeDistances = points.associate { it.name to plane.signedDistance(it.point) }
        val planeRms = sqrt(
            planeDistances.values.map { it * it }.average()
        )

        // Project every point onto the fitted ground plane before measuring.
        val projected = points.associate { it.name to plane.project(it.point) }
        val jackPoint = projected[jack.name] ?: return null

        val result = woods.map { wood ->
            val p = projected[wood.name] ?: return null
            val d = (p - jackPoint).length()
            wood.name to d
        }.sortedBy { it.second }

        val minimum = result.firstOrNull()?.second ?: return null
        if (minimum <= 1e-12) return null

        return PlaneModelResult(
            measurements = result.mapIndexed { index, pair ->
                CorrectedMeasurement(pair.first, pair.second / minimum, index + 1)
            },
            planeRms = planeRms,
            focalLengthPx = focalLengthPx
        )
    }

    private data class Plane(val normal: Vec3, val offset: Double) {
        fun signedDistance(p: Vec3): Double = normal.dot(p) - offset
        fun project(p: Vec3): Vec3 = p - normal * signedDistance(p)
    }

    /** Least-squares plane through the supplied points. */
    private fun fitPlane(points: List<Vec3>): Plane? {
        if (points.size < 3) return null

        // Centroid.
        val centroid = points.reduce { a, b -> a + b } * (1.0 / points.size)

        // Find the least-squares normal by taking the eigenvector associated
        // with the smallest eigenvalue of the 3x3 covariance matrix.
        var a00 = 0.0;
        var a01 = 0.0;
        var a02 = 0.0
        var a11 = 0.0;
        var a12 = 0.0;
        var a22 = 0.0
        for (p in points) {
            val q = p - centroid
            a00 += q.x * q.x
            a01 += q.x * q.y
            a02 += q.x * q.z
            a11 += q.y * q.y
            a12 += q.y * q.z
            a22 += q.z * q.z
        }

        val normal = smallestEigenvector(
            doubleArrayOf(
                a00, a01, a02,
                a01, a11, a12,
                a02, a12, a22
            )
        ) ?: return null

        val n = normal.normalized()
        return Plane(n, n.dot(centroid))
    }

    /** Jacobi diagonalisation for a small symmetric 3x3 matrix. */
    private fun smallestEigenvector(m0: DoubleArray): Vec3? {
        val m = m0.copyOf()
        val v = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )

        repeat(20) {
            var p = 0
            var q = 1
            var max = kotlin.math.abs(m[1])
            if (kotlin.math.abs(m[2]) > max) {
                p = 0; q = 2; max = kotlin.math.abs(m[2])
            }
            if (kotlin.math.abs(m[5]) > max) {
                p = 1; q = 2; max = kotlin.math.abs(m[5])
            }
            if (max < 1e-12) return@repeat

            val pp = m[p * 3 + p]
            val qq = m[q * 3 + q]
            val pq = m[p * 3 + q]
            val phi = 0.5 * kotlin.math.atan2(2.0 * pq, qq - pp)
            val c = kotlin.math.cos(phi)
            val s = kotlin.math.sin(phi)

            for (k in 0..2) {
                val mkp = m[k * 3 + p]
                val mkq = m[k * 3 + q]
                m[k * 3 + p] = c * mkp - s * mkq
                m[k * 3 + q] = s * mkp + c * mkq
            }
            for (k in 0..2) {
                val mpk = m[p * 3 + k]
                val mqk = m[q * 3 + k]
                m[p * 3 + k] = c * mpk - s * mqk
                m[q * 3 + k] = s * mpk + c * mqk
            }
            for (k in 0..2) {
                val vkp = v[k * 3 + p]
                val vkq = v[k * 3 + q]
                v[k * 3 + p] = c * vkp - s * vkq
                v[k * 3 + q] = s * vkp + c * vkq
            }
        }

        val eigenvalues = doubleArrayOf(m[0], m[4], m[8])
        var index = 0
        if (eigenvalues[1] < eigenvalues[index]) index = 1
        if (eigenvalues[2] < eigenvalues[index]) index = 2

        return Vec3(v[index], v[3 + index], v[6 + index])
    }
}
