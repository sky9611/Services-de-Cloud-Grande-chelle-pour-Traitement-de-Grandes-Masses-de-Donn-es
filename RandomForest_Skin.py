import pandas as pd
import numpy as np
import timeit
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import StringIndexer, Bucketizer,OneHotEncoder,VectorAssembler,PCA


sc = SparkContext(appName="PythonRandomForestClassificationForPostures")
sqlContext = SQLContext(sc)

# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, '/home/wth/Downloads/skin_nonskin.txt')

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

start = timeit.default_timer()
# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=10, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

end = timeit.default_timer()
print("Tranining time: ",str(end-start),"s",sep='')

# Evaluate model on test instances and compute test error
start = timeit.default_timer()
predictions = model.predict(testData.map(lambda x: x.features))
end = timeit.default_timer()
print("Prediction time: ",str(end-start),"s",sep='')
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(testData.count())
print('Test Error = ' + str(testErr))
# print('Learned classification forest model:')
# print(model.toDebugString())

sc.stop()
