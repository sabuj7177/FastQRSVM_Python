from pyspark import SparkConf, SparkContext, AccumulatorParam
from pyspark.sql import SparkSession
import math
import numpy as np
from functools import partial

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
optStepSize = (math.floor(1.0 / (learnRate * C)) - 0.5) * learnRate
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
xAcc = sc.accumulator(np.zeros((totalData, 1)), VectorAccumulatorParam())
alphaList = sc.accumulator(np.zeros((totalData, 1)), VectorAccumulatorParam())
betaList = sc.accumulator(np.zeros((totalData, 1)), VectorAccumulatorParam())

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

def LocalQtX(localReflector, subX):
    (rowInCluster,colInCluster) = subX.shape
    for k in range(nCols):
        value = np.matmul(np.transpose(localReflector[0:rowInCluster,k]),subX[0:rowInCluster,0])
        temp = (2*value)*localReflector[0:rowInCluster,k]
        subX[0:rowInCluster,0] = (subX[0:rowInCluster,0]-temp).ravel()
    return subX[0:rowInCluster,0]

def GlobalQtX(globalReflector, subX):
    (rowInReflector,colInReflector) = globalReflector.shape
    for k in range(nCols):
        value = np.matmul(np.transpose(globalReflector[0:rowInReflector,k]),subX[0:rowInReflector,0])
        temp = (2*value)*globalReflector[0:rowInReflector,k]
        subX[0:rowInReflector,0] = (subX[0:rowInReflector,0]-temp).ravel()
    return subX[0:rowInReflector,0]

def LocalQtXPerCluster(num, list_of_lists,fullX, recordCountInClusters):
    partitionedMatrix = np.zeros((numOfPartialData + thresholdSpark, nCols))
    final_iterator = [5]
    currentRow = 0
    for x in list_of_lists:
        partitionedMatrix[currentRow, :] = x
        currentRow += 1
    reflector = partitionedMatrix[0:currentRow, :]
    rowInCluster = int(recordCountInClusters[num][0])
    previousRows = 0
    for x in range(num):
        previousRows += int(recordCountInClusters[x][0])
    subX = fullX[previousRows:previousRows+rowInCluster,:]
    updatedSubX = LocalQtX(reflector,subX)
    temp = np.zeros((totalData, 1))
    temp[previousRows:previousRows+rowInCluster, 0] = updatedSubX.ravel()
    xAcc.add(temp)
    return iter(final_iterator)


def Dist_QtX(x, globalReflector, localReflector, recordCountInClusters):
    dummy = localReflector.mapPartitionsWithIndex(
        partial(LocalQtXPerCluster, fullX=x, recordCountInClusters=recordCountInClusters))
    dummy.count()
    global xAcc
    finalX = xAcc.value
    xAcc = sc.accumulator(np.zeros((totalData, 1)), VectorAccumulatorParam())

    partialX = np.zeros((numOfClusters*nCols, 1))
    fullIterator = 0
    partialIterator = 0
    for k in range(numOfClusters):
        temp = finalX[fullIterator:fullIterator+nCols,0]
        partialX[partialIterator:partialIterator+nCols,0] = temp.ravel()
        fullIterator += int(recordCountInClusters[k][0])
        partialIterator += nCols
    updatedPartialX = GlobalQtX(globalReflector, partialX)
    rowNumberCount = 0
    partialIterator = 0
    for k in range(numOfClusters):
        finalX[rowNumberCount:rowNumberCount+nCols,0] = updatedPartialX[partialIterator:partialIterator+nCols].ravel()
        rowNumberCount += int(recordCountInClusters[k][0])
        partialIterator += nCols
    return finalX

def LocalQX(localReflector, subX):
    (rowInCluster,colInCluster) = subX.shape
    for k in range(nCols-1, -1,-1):
        value = np.matmul(np.transpose(localReflector[0:rowInCluster,k]),subX[0:rowInCluster,0])
        temp = (2*value)*localReflector[0:rowInCluster,k]
        subX[0:rowInCluster,0] = (subX[0:rowInCluster,0]-temp).ravel()
    return subX[0:rowInCluster,0]

def GlobalQX(globalReflector, subX):
    (rowInReflector,colInReflector) = globalReflector.shape
    for k in range(nCols-1, -1,-1):
        value = np.matmul(np.transpose(globalReflector[0:rowInReflector,k]),subX[0:rowInReflector,0])
        temp = (2*value)*globalReflector[0:rowInReflector,k]
        subX[0:rowInReflector,0] = (subX[0:rowInReflector,0]-temp).ravel()
    return subX[0:rowInReflector,0]

def LocalQXPerCluster(num, list_of_lists,fullX, recordCountInClusters):
    partitionedMatrix = np.zeros((numOfPartialData + thresholdSpark, nCols))
    final_iterator = [5]
    currentRow = 0
    for x in list_of_lists:
        partitionedMatrix[currentRow, :] = x
        currentRow += 1
    reflector = partitionedMatrix[0:currentRow, :]
    rowInCluster = int(recordCountInClusters[num][0])
    previousRows = 0
    for x in range(num):
        previousRows += int(recordCountInClusters[x][0])
    subX = fullX[previousRows:previousRows+rowInCluster,:]
    updatedSubX = LocalQX(reflector,subX)
    temp = np.zeros((totalData, 1))
    temp[previousRows:previousRows+rowInCluster, 0] = updatedSubX.ravel()
    xAcc.add(temp)
    return iter(final_iterator)


def Dist_QX(x, globalReflector, localReflector, recordCountInClusters):
    partialX = np.zeros((numOfClusters*nCols, 1))
    fullIterator = 0
    partialIterator = 0
    for k in range(numOfClusters):
        temp = x[fullIterator:fullIterator+nCols,0]
        partialX[partialIterator:partialIterator+nCols,0] = temp.ravel()
        fullIterator += int(recordCountInClusters[k][0])
        partialIterator += nCols
    updatedPartialX = GlobalQX(globalReflector, partialX)
    rowNumberCount = 0
    partialIterator = 0
    for k in range(numOfClusters):
        x[rowNumberCount:rowNumberCount+nCols,0] = updatedPartialX[partialIterator:partialIterator+nCols].ravel()
        rowNumberCount += int(recordCountInClusters[k][0])
        partialIterator += nCols
    dummy = localReflector.mapPartitionsWithIndex(partial(LocalQXPerCluster,fullX = x,recordCountInClusters = recordCountInClusters))
    dummy.count()
    global xAcc
    updatedX = xAcc.value
    xAcc = sc.accumulator(np.zeros((totalData, 1)), VectorAccumulatorParam())
    return updatedX

def AlphaBetaUpdate(F, betaCapOld, Ecap):
    alphaCap = np.matmul(np.linalg.inv(F),Ecap-betaCapOld)
    betaCap = betaCapOld - (optStepSize * alphaCap)
    return alphaCap, betaCap

def AlphaBetaUpdatePerCluster(num, list_of_lists,betaBroadcast, enBroadcast, recordCountInClusters):
    for x in list_of_lists:
        F = x
    #partitionedMatrix = np.zeros((numOfPartialData + thresholdSpark, nCols))
    final_iterator = [5]
    rowInCluster = int(recordCountInClusters[num][0])
    previousRows = 0
    for x in range(num):
        previousRows += int(recordCountInClusters[x][0])
    fullBeta = betaBroadcast.value
    fullEn = enBroadcast.value
    subBeta = fullBeta[previousRows:previousRows+rowInCluster,:]
    subEn = fullEn[previousRows:previousRows+rowInCluster,:]
    alphaCap, betaCap = AlphaBetaUpdate(F, subBeta, subEn)
    temp = np.zeros((totalData, 1))
    temp[previousRows:previousRows+rowInCluster, 0] = alphaCap.ravel()
    alphaList.add(temp)
    temp = np.zeros((totalData, 1))
    temp[previousRows:previousRows+rowInCluster, 0] = betaCap.ravel()
    betaList.add(temp)
    return iter(final_iterator)

def QRDecompositionPerCluster(num, list_of_lists):
    partitionedMatrix = np.zeros((numOfPartialData+thresholdSpark,nCols))
    #final_iterator = [5]
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

def TestData(alphaTotalMat, finalR, testDataMat, testLabelMat):
    weightMat = np.matmul(np.transpose(finalR), alphaTotalMat[0:nCols,:])
    transposeWeightMat = np.transpose(weightMat)
    result = np.matmul(transposeWeightMat,np.transpose(testDataMat))
    correct = 0
    wrong = 0
    correntPos = 0
    correntNeg = 0
    wrongPos = 0
    wrongNeg = 0
    for i in range(testrow):
        if result[0][i] >0:
            if testLabelMat[i][0] == testLabelPositive:
                correct += 1
                correntPos += 1
            else:
                wrong += 1
                wrongPos += 1
        else:
            if testLabelMat[i][0] == testLabelNegative:
                correct += 1
                correntNeg += 1
            else:
                wrong += 1
                wrongNeg += 1

    accuracy = (correct *100)/ testrow
    print("corrent is :" + str(accuracy))
    print("wrong is :" + str(accuracy))
    print("Accuracy is :" + str(accuracy))


def main():
    #sc = SparkContext(master="local", appName="meka")
    spark = SparkSession(sc)

    #global numOfPartialData
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
    localReflectors = trainingData.mapPartitionsWithIndex(QRDecompositionPerCluster)
    localReflectors.count()
    #print(recordCountInClusters.value)
    #print(Rgatherac.value)
    RgatherMat = Rgatherac.value
    Q, finalR, globalReflector = QR_Factorization(RgatherMat, numOfClusters*nCols, nCols)
    recordCountInClustersMat = recordCountInClusters.value

    betaCapmat = np.full((totalData,1),1.0)
    betaBroadCast = sc.broadcast(betaCapmat)

    Ecap = Dist_QtX(np.full((totalData,1),-1.0),globalReflector,localReflectors,recordCountInClustersMat)
    enBroadCast = sc.broadcast(Ecap)

    fullRfinalTransposeMat = np.transpose(finalR)
    RgRgTmat = np.matmul(finalR,fullRfinalTransposeMat)
    Fs = []
    F = np.zeros((int(recordCountInClustersMat[0][0]), int(recordCountInClustersMat[0][0])))
    F[0:nCols,0:nCols] = (-1)*RgRgTmat
    for i in range(int(recordCountInClustersMat[0][0])):
        F[i][i] += (-1.0/(2*C))
    Fs.append(F)
    for p in range(numOfClusters-1):
        F = np.zeros((int(recordCountInClustersMat[p+1][0]), int(recordCountInClustersMat[p+1][0])))
        for i in range(int(recordCountInClustersMat[p+1][0])):
            F[i][i] += (-1.0 / (2 * C))
        Fs.append(F)

    FsRDD = sc.parallelize(Fs, numOfClusters)
    prevBetaCap = np.full((totalData,1),1.0)

    for it in range(maxIteration): 
        global alphaList
        alphaList = sc.accumulator(np.zeros((totalData, 1)), VectorAccumulatorParam())
        global betaList
        betaList = sc.accumulator(np.zeros((totalData, 1)), VectorAccumulatorParam())
        dummy = FsRDD.mapPartitionsWithIndex(partial(AlphaBetaUpdatePerCluster, betaBroadcast = betaBroadCast, enBroadcast = enBroadCast, recordCountInClusters = recordCountInClustersMat))
        dummy.count()
        Betamat = Dist_QX(betaList.value, globalReflector, localReflectors, recordCountInClustersMat)
        for i in range(totalData):
            if Betamat[i][0] < 0:
                Betamat[i][0] = 0
        betaCapmat = Dist_QtX(Betamat, globalReflector, localReflectors, recordCountInClustersMat)
        diffBeta = betaCapmat - prevBetaCap
        error = np.linalg.norm(diffBeta,ord=1)
        prevBetaCap = betaCapmat
        print("####################Here is Iteration : "+str(it)+" ############################")
        print("This is error :"+str(error))
        betaBroadCast = sc.broadcast(betaCapmat)
        if it% 5 == 0:
            TestData(alphaList.value, finalR, testdatamat, testlabelmat)
        if error<threshold:
            break

    TestData(alphaList.value, finalR, testdatamat, testlabelmat)

    '''dummyMap = filtered_lists.map(dummyFunc)
    print(dummyMap.collect())
    dummyReduce = dummyMap.reduce(lambda a, b: a + b)
    print(dummyReduce)'''



main()