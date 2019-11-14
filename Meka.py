from pyspark import SparkConf,SparkContext

filename ="a9a/a9a_train_data.txt"
sc = SparkContext(master="local",appName="meka")
print(sc.textFile(filename).first())

words = sc.parallelize (
   ["scala",
   "java",
   "hadoop",
   "spark",
   "akka",
   "spark vs hadoop",
   "pyspark",
   "pyspark and spark"]
)

counts = words.count()
print "Number of elements in RDD -> %i" % (counts)

coll = words.collect()
print "Elements in RDD -> %s" % (coll)