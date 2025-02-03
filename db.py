from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SurvivalAnalysis").getOrCreate()
