"""
Spark Streaming Job for Real-time Analytics
Processes Kafka messages and writes to Elasticsearch
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, window, count, avg, 
    sum as spark_sum, max as spark_max
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, TimestampType
)

# Define schema for telemetry data
telemetry_schema = StructType([
    StructField("device_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("demographics", StructType([
        StructField("age", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("emotion", StringType(), True)
    ]), True),
    StructField("confidence", DoubleType(), True),
    StructField("event_type", StringType(), True)
])


def create_spark_session():
    """Create Spark session with Elasticsearch connector"""
    return SparkSession.builder \
        .appName("RetailAnalyticsStreaming") \
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0") \
        .getOrCreate()


def process_streaming_data():
    """Main streaming processing function"""
    spark = create_spark_session()
    
    # Read from Kafka
    kafka_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "retail-kafka-kafka-bootstrap:9092") \
        .option("subscribe", "edge-telemetry") \
        .option("startingOffsets", "latest") \
        .load()
    
    # Parse JSON
    telemetry_df = kafka_df.select(
        from_json(col("value").cast("string"), telemetry_schema).alias("data")
    ).select("data.*")
    
    # Extract nested fields
    analytics_df = telemetry_df.select(
        col("device_id"),
        col("timestamp").cast("timestamp"),
        col("demographics.age").alias("age"),
        col("demographics.gender").alias("gender"),
        col("demographics.emotion").alias("emotion"),
        col("confidence")
    )
    
    # Aggregate by time windows
    windowed_stats = analytics_df \
        .withWatermark("timestamp", "1 minute") \
        .groupBy(
            window(col("timestamp"), "5 minutes"),
            col("device_id")
        ) \
        .agg(
            count("*").alias("total_customers"),
            avg("age").alias("avg_age"),
            spark_sum(
                (col("emotion") == "Happy").cast("int"))
            ).alias("happy_count"),
            spark_sum(
                (col("gender") == "Female").cast("int"))
            ).alias("female_count")
        )
    
    # Write to Elasticsearch
    def write_to_elasticsearch(batch_df, batch_id):
        """Write batch to Elasticsearch"""
        batch_df.write \
            .format("org.elasticsearch.spark.sql") \
            .option("es.nodes", "elasticsearch") \
            .option("es.port", "9200") \
            .option("es.resource", "retail-analytics/metrics") \
            .option("es.mapping.id", "device_id") \
            .mode("append") \
            .save()
    
    # Start streaming query
    query = windowed_stats.writeStream \
        .foreachBatch(write_to_elasticsearch) \
        .outputMode("update") \
        .option("checkpointLocation", "/tmp/checkpoint/analytics") \
        .start()
    
    query.awaitTermination()


if __name__ == "__main__":
    process_streaming_data()

