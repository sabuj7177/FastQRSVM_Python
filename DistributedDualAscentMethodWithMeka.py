from pyspark import SparkConf, SparkContext, AccumulatorParam
from pyspark.sql import SparkSession
import math
import numpy as np

# filename = "a9a_1000/a9a_train_1000_40_with_label.txt"
# sc = SparkContext(master="local",appName="meka")
# print(sc.textFile(filename).first())
conf = SparkConf().setAppName('appName').setMaster('local')
sc = SparkContext(conf=conf)
numOfClusters = 4
totalData = 1000
testrow = 1000
nCols = 40
learnRate = 0.2
C = 1.0
threshold = 0.001
maxIteration = 100
numOfPartialData = int(math.ceil(totalData/numOfClusters))
thresholdSpark = 10
trainDataPath = "a9a_1000/a9a_train_1000_40_with_label.txt"
testDataPath = "a9a_1000/a9a_test_1000_40.txt"
testLabelPath = "a9a_1000/a9a_test_Label_1000.txt"
testLabelPositive = 1.0 #Positive label of the data set
testLabelNegative = -1.0 #Negative label of the data set
rowNumberCount = 0
#Accumulator Defination

class VectorAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return initialValue

    def addInPlace(self, v1, v2):
        #v1 += v2
        return np.add(v1,v2)

#Accumulator Definition Completes

recordCountInClusters = sc.accumulator(np.zeros((numOfClusters,1)), VectorAccumulatorParam())
Rgatherac = sc.accumulator(np.zeros((numOfClusters*nCols,nCols)), VectorAccumulatorParam())

# This part is for HouseHolder QRDecomposition
def column_convertor(x):
    """
    Converts 1d array to column vector
    """
    x.shape = (1, x.shape[0])
    return x


def get_norm(x):
    """
    Returns Norm of vector x
    """
    return np.sqrt(np.sum(np.square(x)))


def householder_transformation(v):
    """
    Returns Householder matrix for vector v
    """
    size_of_v = v.shape[1]
    e1 = np.zeros_like(v)
    e1[0, 0] = 1
    vector = get_norm(v) * e1
    if v[0, 0] < 0:
        vector = - vector
    u = (v + vector).astype(np.float32)
    norm2 = get_norm(u)
    u = u / norm2
    H = np.identity(size_of_v) - ((2 * np.matmul(np.transpose(u), u)) / np.matmul(u, np.transpose(u)))
    return H, u


def qr_step_factorization(q, r, iter, n):
    """
    Return Q and R matrices for iter number of iterations.
    """
    v = column_convertor(r[iter:, iter])
    Hbar, reflect = householder_transformation(v)
    H = np.identity(n)
    H[iter:, iter:] = Hbar
    r = np.matmul(H, r)
    q = np.matmul(q, H)
    return q, r,reflect

def QR_Factorization(A,n,m):
    Q = np.identity(n)
    R = A.astype(np.float32)
    Reflectors = np.zeros((n, m))
    for i in range(min(n, m)):
        # For each iteration, H matrix is calculated for (i+1)th row
        Q, R, reflect = qr_step_factorization(Q, R, i, n)
        Reflectors[i:n, i] = np.transpose(reflect).ravel()
    min_dim = min(m, n)
    R = np.around(R, decimals=6)
    R = R[:min_dim, :min_dim]
    Q = np.around(Q, decimals=6)
    return Q,R,Reflectors

#HouseHolder QRDecomposition ends here



def inputFuncTrain(dataLine):
    X = np.zeros((1, nCols))
    dataLine = dataLine.strip()
    splitBySpace = dataLine.split(' ')
    for j in range(nCols):
        X[0, j] = float(splitBySpace[j])
    return X

def inputFuncTest(dataLine):
    X = np.zeros((1, nCols))
    dataLine = dataLine.strip()
    splitByComma = dataLine.split(',')
    for j in range(nCols):
        X[0, j] = float(splitByComma[j])
    return X

def inputFuncTestLabel(dataLine):
    X = np.zeros((1, 1))
    dataLine = dataLine.strip()
    splitBySpace = dataLine.split(' ')
    X[0, 0] = float(splitBySpace[0])
    return X

def filter_out_2_from_partition(num, list_of_lists):
    partitionedMatrix = np.zeros((numOfPartialData+thresholdSpark,nCols))
    final_iterator = [5]
    currentRow = 0
    for x in list_of_lists:
        partitionedMatrix[currentRow,:] = x
        currentRow += 1
    trimmedMatrix = partitionedMatrix[0:currentRow,:]
    Q, R, Reflectors = QR_Factorization(trimmedMatrix,currentRow,nCols)
    temp = np.zeros((numOfClusters,1))
    temp[num,0] = currentRow
    recordCountInClusters.add(temp)
    temp = np.zeros((numOfClusters*nCols,nCols))
    temp[num*nCols:(num+1)*nCols,:] = R
    Rgatherac.add(temp)
    return iter(Reflectors)

def dummyFunc(s):
    return s

def main():
    #sc = SparkContext(master="local", appName="meka")




    spark = SparkSession(sc)

    #global numOfPartialData
    optStepSize = (math.floor(1.0 / (learnRate * C)) - 0.5) * learnRate
    #numOfPartialData = math.ceil(totalData/numOfClusters)
    #print(numOfPartialData)

    trainingDataRDD = sc.textFile(trainDataPath,numOfClusters)
    trainingData = trainingDataRDD.map(inputFuncTrain).persist()

    testDataRDD = sc.textFile(testDataPath, numOfClusters)
    testData = testDataRDD.map(inputFuncTest).persist()

    testLabelRDD = sc.textFile(testLabelPath, numOfClusters)
    testLabel = testLabelRDD.map(inputFuncTestLabel).persist()

    testdatamat = np.zeros((testrow, nCols))
    testlabelmat = np.zeros((testrow, 1))

    count = 0
    for x in testData.take(testrow):
        for j in range(nCols):
            testdatamat[count,j] = x[0,j]
        count+=1

    count = 0
    for y in testLabel.take(testrow):
        testlabelmat[count, 0] = y[0, 0]
        count += 1

    #trainingData.foreachPartition(filter_out_2_from_partition)

    # rdd = sc.parallelize(range(1, 4)).map(lambda x: (x, "a" * x))
    # print(rdd.collect())
    filtered_lists = trainingData.mapPartitionsWithIndex(filter_out_2_from_partition)
    filtered_lists.count()
    print(recordCountInClusters.value)
    print(Rgatherac.value)
    '''dummyMap = filtered_lists.map(dummyFunc)
    print(dummyMap.collect())
    dummyReduce = dummyMap.reduce(lambda a, b: a + b)
    print(dummyReduce)'''



main()