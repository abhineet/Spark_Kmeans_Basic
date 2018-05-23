
"""Basic Example of using 
Kmeans clustering using pyspark"""

## Imports

from pyspark import SparkConf, SparkContext

from operator import add
import sys
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
#Other Version of Kmeans 
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql import SparkSession


## Constants
APP_NAME = "Kmeans Example"
##OTHER FUNCTIONS/CLASSES

def main(spark,filename):
   df = spark.read.csv(filename,header=True,inferSchema=True)
#    >>> df.show(4)
# +----+----+----+
# |col1|col2|col3|
# +----+----+----+
# |   7|   4|   1|
# |   7|   7|   9|
# |   7|   9|   6|
# |   1|   6|   5|
# +----+----+----+
   vector_assembler = VectorAssembler(inputCols=['col1','col2','col3'],outputCol='features')
   v_cluster_df = vector_assembler.transform(df)
#    >>> v_cluster_df.show(4)
# +----+----+----+-------------+
# |col1|col2|col3|     features|
# +----+----+----+-------------+
# |   7|   4|   1|[7.0,4.0,1.0]|
# |   7|   7|   9|[7.0,7.0,9.0]|
# |   7|   9|   6|[7.0,9.0,6.0]|
# |   1|   6|   5|[1.0,6.0,5.0]|
# +----+----+----+-------------+
# >>> v_cluster_df.printSchema()
# root
#  |-- col1: integer (nullable = true)
#  |-- col2: integer (nullable = true)
#  |-- col3: integer (nullable = true)
#  |-- features: vector (nullable = true)
   kmeans = KMeans().setK(3)
   kmeans = kmeans.setSeed(1)
   kmodel = kmeans.fit(v_cluster_df)
   centres = kmodel.clusterCenters()
   print(centres)
   # [array([ 35.88461538,  31.46153846,  34.42307692]), array([ 5.12,  5.84,  4.84]), array([ 80.        ,  79.20833333,  78.29166667])]
   bkmeans = BisectingKMeans().setK(3)
   bkmeans = bkmeans.setSeed(1)
   bkmodel = bkmeans.fit(v_cluster_df)
   bcentres = bkmodel.clusterCenters()
   print(bcentres)

if __name__ == "__main__":

   # Configure Spark
   # conf = SparkConf().setAppName(APP_NAME)
   # conf = conf.setMaster("local[*]")
   # sc   = SparkContext(conf=conf)
   filename = sys.argv[1]
   spark = SparkSession\
        .builder\
        .appName(APP_NAME)\
        .getOrCreate()
   # Execute Main functionality
   main(spark, filename)