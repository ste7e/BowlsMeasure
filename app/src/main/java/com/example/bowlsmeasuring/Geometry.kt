package com.example.bowlsmeasuring

data class Point2(val x: Float, val y: Float)
data class ObjectMeasurement(val name: String, val centre: Point2, val apparentRadiusPx: Float)
data class CorrectedMeasurement(val name: String, val relativeDistance: Float, val rank: Int)

object SizeDepthModel {
    fun calculate(
        jack: ObjectMeasurement,
        jackToWoodDiameterRatio: Float,
        focalLengthPx: Double,
        principalX: Double,
        principalY: Double,
        woods: List<ObjectMeasurement>
    ): List<CorrectedMeasurement> {
        if (woods.isEmpty()) return emptyList()
        fun cameraPoint(
            o: ObjectMeasurement, equivalentWoodRadius: Double
        ): Triple<Double, Double, Double> {
            val z = 1.0 / equivalentWoodRadius
            val x = (o.centre.x.toDouble() - principalX) * z / focalLengthPx
            val y = (o.centre.y.toDouble() - principalY) * z / focalLengthPx
            return Triple(x, y, z)
        }

        val j = cameraPoint(
            jack, jack.apparentRadiusPx.toDouble() / jackToWoodDiameterRatio
        )

        val result = woods.map { w ->
            val p = cameraPoint(w, w.apparentRadiusPx.toDouble())
            val dx = p.first - j.first;
            val dy = p.second - j.second;
            val dz = p.third - j.third
            w.name to kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
        }.sortedBy { it.second }
        val minimum = result.first().second
        return result.mapIndexed { i, pair ->
            CorrectedMeasurement(
                pair.first, (pair.second / minimum).toFloat(), i + 1
            )
        }
    }
}
