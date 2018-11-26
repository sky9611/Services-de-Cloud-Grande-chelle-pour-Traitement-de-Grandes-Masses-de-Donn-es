from pyspark import SparkContext, SparkConf
from pyspark.ml import Pipeline
from pyspark.mllib.util import MLUtils
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, Bucketizer,OneHotEncoder,VectorAssembler,PCA
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load the data stored in LIBSVM format as a DataFrame.
sqlContext = SQLContext(sc)
data = MLUtils.loadLibSVMFile(sc, '/home/wth/Downloads/skin_nonskin.txt')

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

start = timeit.default_timer()

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

end = timeit.default_timer()
print("Tranining time: ",str(end-start),"s",sep='')

# Make predictions.
start = timeit.default_timer()
predictions = model.transform(testData)
end = timeit.default_timer()
print("Prediction time: ",str(end-start),"s",sep='')

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[2]
# summary only
print(treeModel)