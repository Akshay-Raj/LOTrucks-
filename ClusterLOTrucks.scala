import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans

object ClusterLOTrucks {

  case class LOTrucks(dt: String, lat: Double, lon: Double, base: String) extends Serializable

  def main(args: Array[String]) {

    val spark: SparkSession = SparkSession.builder().appName("LOTrucks").getOrCreate()

    import spark.implicits._

    val schema = StructType(Array(
      StructField("dt", TimestampType, true),
      StructField("lat", DoubleType, true),
      StructField("lon", DoubleType, true),
      StructField("base", StringType, true)
    ))

    // Spark 2.1
    val df: Dataset[LOTrucks] = spark.read.option("inferSchema", "false").schema(schema).csv("/LOTrucks.csv").as[LOTrucks]

    df.cache
    df.show
    df.schema

    val featureCols = Array("lat", "lon")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(df)
    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), 5043)

    // increase the iterations if running on a cluster (this runs on a 1 node sandbox)
    val kmeans = new KMeans().setK(20).setFeaturesCol("features").setMaxIter(5)
    val model = kmeans.fit(trainingData)
    println("Final Centers: ")
    model.clusterCenters.foreach(println)

    val categories = model.transform(testData)

    categories.show
    categories.createOrReplaceTempView("LOTrucks")

    categories.select(month($"dt").alias("month"), dayofmonth($"dt").alias("day"), hour($"dt").alias("hour"), $"prediction").groupBy("month", "day", "hour", "prediction").agg(count("prediction").alias("count")).orderBy("day", "hour", "prediction").show

    categories.select(hour($"dt").alias("hour"), $"prediction").groupBy("hour", "prediction").agg(count("prediction")
      .alias("count")).orderBy(desc("count")).show

    categories.groupBy("prediction").count().show()

    spark.sql("select prediction, count(prediction) as count from LOTrucks group by prediction").show

    spark.sql("SELECT hour(LOTrucks.dt) as hr,count(prediction) as ct FROM LOTrucks group By hour(LOTrucks.dt)").show

    /*
     * uncomment below for various functionality:
    */
    // to save the model
    model.write.overwrite().save("/savemodel")
    // model can be  re-loaded like this
    // val sameModel = KMeansModel.load("/savemodel")
    //
    // to save the categories dataframe as json data
    val res = spark.sql("select dt, lat, lon, base, prediction as cid FROM LOTrucks order by dt")
    res.write.format("json").save("/LOTrucks.json")
  }
}
