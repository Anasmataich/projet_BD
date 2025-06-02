from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, concat, lit
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, Normalizer
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder.appName("Food Recommender with NLP").getOrCreate()

df = spark.read.json("data/en.openfoodfacts.org.products.json")

df = df.withColumn("url", concat(lit("https://world.openfoodfacts.org/product/"), col("code")))

df_clean = df.select(
    "product_name",
    "brands",
    "ingredients_text",
    "nutrition_grade_fr",
    "countries_tags",
    "allergens",
    "image_url",
    "url"
).filter(
    (col("product_name").isNotNull()) & 
    (col("ingredients_text").isNotNull())
)

df_nlp = df_clean.limit(5000)

#  NLP Pipeline
tokenizer = Tokenizer(inputCol="ingredients_text", outputCol="words")
wordsData = tokenizer.transform(df_nlp)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)
featurizedData = hashingTF.transform(wordsData)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

normalizer = Normalizer(inputCol="features", outputCol="normFeatures")
normalizedData = normalizer.transform(rescaledData)

#  KMeans Clustering
kmeans = KMeans(featuresCol="normFeatures", predictionCol="cluster", k=5, seed=42)
model = kmeans.fit(normalizedData)
clusteredData = model.transform(normalizedData)

def advanced_recommend(country=None, ingredients=None, nutriscore_min=None, nutriscore_max=None, allergen_free=False, top_n=20):
    df_filtered = clusteredData

    if country:
        df_filtered = df_filtered.filter(lower(col("countries_tags")).like(f"%{country.lower()}%"))
    if ingredients:
        df_filtered = df_filtered.filter(lower(col("ingredients_text")).like(f"%{ingredients.lower()}%"))
    if nutriscore_min:
        df_filtered = df_filtered.filter(col("nutrition_grade_fr") >= nutriscore_min.lower())
    if nutriscore_max:
        df_filtered = df_filtered.filter(col("nutrition_grade_fr") <= nutriscore_max.lower())
    if allergen_free:
        df_filtered = df_filtered.filter((col("allergens").isNull()) | (col("allergens") == ""))

    results = df_filtered.select(
        "product_name", "brands", "nutrition_grade_fr",
        "ingredients_text", "image_url", "url", "cluster"
    ).orderBy("cluster").limit(top_n).collect()

    return [{
        "product_name": row["product_name"] or "Inconnu",
        "brands": row["brands"] or "N/A",
        "nutrition_grade_fr": row["nutrition_grade_fr"] or "N/A",
        "ingredients_text": row["ingredients_text"] or "N/A",
        "cluster": row["cluster"],
        "image_url": row["image_url"] or None,
        "url": row["url"] or "https://world.openfoodfacts.org"
    } for row in results]

