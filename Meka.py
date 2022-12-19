'''from pyspark import SparkConf,SparkContext

filename = "a9a/a9a_train_data.txt"
sc = SparkContext(master="local",appName="meka")
print(sc.textFile(filename).first())'''

import numpy as np
import random
import math
from scipy.linalg import svd
from sklearn.metrics.pairwise import euclidean_distances
from numpy import linalg as LA
from sklearn.cluster import KMeans
MAXDIS = 1000000000000000.0
tol = 0.0000000001

def KMeansClustering(X, clusterNum, maxIter, colNum, taken):
    totalWeight = taken
    idx = np.full((taken),-1)
    '''print("Starting xnorm")
    #centers = np.zeros((clusterNum,colNum))
    xnorm = np.zeros((taken))
    for i in range(taken):
        for j in range(colNum):
            xnorm[i] += X[i,j]*X[i,j]
    print(xnorm)'''
    xnorm = np.square(LA.norm(X,axis=1))
    #print("After library")
    #print(xnorm)

    for i in range(taken):
        idx[i] = random.randint(0,clusterNum-1)

    print("Kmeans: before iteration")

    for iter in range(maxIter):
        centers = np.zeros((clusterNum,colNum))
        count = np.zeros((clusterNum))
        for i in range(taken):
            for j in range(colNum):
                centers[idx[i],j] += X[i,j]
            count[idx[i]] += 1

        '''centers2 = np.copy(centers)
        for i in range(clusterNum):
            for j in range(colNum):
                centers[i,j] /= count[i]
        print(centers)'''
        #Get the median of each cluster
        centers = centers/count[:,None]
        #print(centers)

        '''centernorm = np.zeros((clusterNum))
        for i in range(clusterNum):
            for j in range(colNum):
                centernorm[i] += centers[i,j]*centers[i,j]
        print(centernorm)'''
        #Get the norm of center
        centernorm = np.square(LA.norm(centers, axis=1))
        #print(centernorm)

        loss = 0
        change = 0
        for i in range(taken):
            currMaxDis = MAXDIS
            curridx = -1
            for k in range(clusterNum):
                currDis = 0
                for j in range(colNum):
                    currDis += X[i,j]*centers[k,j]
                currDis = xnorm[i] - 2*currDis + centernorm[k]
                if currDis<currMaxDis:
                    currMaxDis = currDis
                    curridx = k

            if idx[i]!= curridx:
                idx[i] = curridx
                change += 1

            loss += currMaxDis

        centerCount = np.zeros((clusterNum))
        for i in range(taken):
            centerCount[idx[i]] += 1

        for k in range(clusterNum):
            if centerCount[k]==0:
                while 1:
                    randData = random.randint(0,taken)
                    if centerCount[idx[randData]]>1:
                        centerCount[idx[randData]] -= 1
                        centerCount[k] += 1
                        idx[randData] = k
                        break

        print("iteration "+str(iter))

        if loss<totalWeight*tol:
            break

    centers = np.zeros((clusterNum,colNum))
    count = np.zeros((clusterNum))
    for i in range(taken):
        for j in range(colNum):
            centers[idx[i],j] += X[i,j]
        count[idx[i]] += 1

    for i in range(clusterNum):
        for j in range(colNum):
            centers[i,j] /= count[i]

    return idx, centers

def NysTrain(Y, m, ki, gamma):
    n = len(Y)
    m = int(m)
    randPerm = np.random.permutation(n)
    #randPerm = list(range(n))
    takenIndex = randPerm[:m]
    center = Y[np.ix_(takenIndex)]
    W = squareDist(center,center,gamma=gamma)
    E = squareDist(Y,center,gamma=gamma)
    U, S, V = svd(W)
    S = pow(S,-0.5)
    ki = int(ki)
    S = np.diag(S[:ki])
    return np.dot(E, np.dot(U[:,:ki], S))

def squareDist(X, Y, gamma):
    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    np.exp(K, K)  # exponentiate K in-place
    return K

def main():
    filename = "realsim"
    clusterNum = 4
    maxIter = 10
    dataNum = 72309
    colNum = 20958
    taken = 10000
    targetRank = 600
    gamma = 0.1
    eta = 0.1

    file1 = open(filename, "r+")
    count = 0
    X = np.zeros((taken, colNum))
    y = np.zeros((taken, 1))
    #randPerm = list(range(dataNum))
    randPerm = np.random.permutation(dataNum)
    takenIndex = randPerm[:taken]

    for i in range(dataNum):
        dataLine = file1.readline()
        if i in takenIndex:
            dataLine = dataLine.strip()
            splitBySpace = dataLine.split(' ')
            y[count, 0] = float(splitBySpace[0])
            dataLen = len(splitBySpace)
            for j in range(dataLen - 1):
                temp = splitBySpace[j + 1]
                splitByColon = temp.split(':')
                X[count, int(splitByColon[0]) - 1] = float(splitByColon[1])
            count += 1

    print("Input done")
    #idx, centers = KMeans(X,clusterNum,maxIter,colNum,taken)

    kmeans = KMeans(n_clusters=clusterNum, max_iter= maxIter, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    print("Clustering done")

    new_idx = np.zeros((taken))
    new_dis = np.zeros((taken,clusterNum))
    for i in range(taken):
        for j in range(clusterNum):
            new_dis[i,j] = np.linalg.norm(X[i]-centers[j])
    new_dis = np.square(new_dis)
    for i in range(taken):
        A = new_dis[i].tolist()
        new_idx[i] = A.index(min(A))
    #new_idx = new_dis.min(1)
    #new_idx = new_idx.flatten()
    #print(new_idx)

    dis = squareDist(centers,centers, gamma=gamma)
    sortedDis = np.sort(dis, axis=None)
    mm = math.ceil((clusterNum*clusterNum-clusterNum)*eta)
    if mm>0:
        threshold = sortedDis[mm-1]
    else:
        threshold = 0

    lst = []
    for k in range(clusterNum):
        indices = [i for i, x in enumerate(new_idx.tolist()) if x == k]
        lst.append(indices)

    indexes = np.array(lst)
    lengths = np.zeros((clusterNum))
    ranklist = np.zeros((clusterNum))
    sumRankList = np.zeros((clusterNum))
    sumRank = 0
    for k in range(clusterNum):
        lengths[k] = len(indexes[k])
        ranklist[k] = math.ceil((targetRank*lengths[k])/taken)
        sumRankList[k] = sumRank
        sumRank += ranklist[k]

    if sumRank!=targetRank:
        ranklist[clusterNum-1] -= (sumRank-targetRank)

    S = np.identity(targetRank, dtype=float)
    Ulist = []
    for k in range(clusterNum):
        ki = min(ranklist[k], lengths[k])
        ranklist[k] = ki
        m = min(2*ki, lengths[k])
        Y = X[np.ix_(indexes[k])]
        U = NysTrain(Y, m, ki, gamma)
        Ulist.append(U)


    for i in range(clusterNum):
        for j in range(i+1,clusterNum):
            if dis[i,j] >= threshold:
                #randPermi = list(range(int(lengths[i])))
                randPermi = np.random.permutation(int(lengths[i]))
                numi = min(4*int(ranklist[i]),int(lengths[i]))
                randi = randPermi[:numi]
                #randPermj = list(range(int(lengths[j])))
                randPermj = np.random.permutation(int(lengths[j]))
                numj = min(4 * int(ranklist[j]), int(lengths[j]))
                randj = randPermj[:numj]
                Ui = Ulist[i][np.ix_(randi)]
                Uj = Ulist[j][np.ix_(randj)]
                res_list_i = [indexes[i][k] for k in randi]
                Ai = X[np.ix_(res_list_i)]
                res_list_j = [indexes[j][k] for k in randj]
                Aj = X[np.ix_(res_list_j)]
                tmpK = squareDist(Ai, Aj, gamma=gamma)
                Z1 = np.dot(np.linalg.pinv(np.dot(Ui.T,Ui), rcond=1e-6),Ui.T)
                Z2 = np.dot(Uj, np.linalg.pinv(np.dot(Uj.T, Uj), rcond=1e-6))
                Z = np.dot(Z1, np.dot(tmpK,Z2))
                si = int(sumRankList[i])
                sj = int(sumRankList[j])
                r = len(Z)
                c = len(Z[0])
                S[si:si+r, sj:sj+c] = Z
                S[sj:sj+c, si:si+r] = Z.T

    finalU = np.zeros((taken, targetRank))
    #r = 0
    c = 0
    for k in range(clusterNum):
        U = Ulist[k]
        for i in range(int(lengths[k])):
            ind = indexes[k][i]
            finalU[ind:ind+1,c:c+len(U[0])] = U[i,:]
        #finalU[r:r+len(U),c:c+len(U[0])] = U
        #r += len(U)
        c += len(U[0])

    #np.savetxt("a9a/finalU.txt", S, fmt="%0.8f")

    print("Training complete")

    print("Start testing")
    #randPerm = list(range(taken))
    randPerm = np.random.permutation(taken)
    testNum = 1000
    testIndex = randPerm[:testNum]
    testData = X[np.ix_(testIndex)]
    tmpK = squareDist(testData,X,gamma=gamma)
    U_samp = finalU[np.ix_(testIndex)]
    err = np.linalg.norm(tmpK-(np.dot(U_samp,np.dot(S,finalU.T))))/np.linalg.norm(tmpK)
    print("Error is "+str(err))
    SU, SS, SV = svd(S)
    SS = pow(SS, -0.5)
    SS = np.diag(SS)
    dataX = np.dot(finalU, np.dot(SU, SS))
    print("Shape of dataX "+str(dataX.shape))
    np.savetxt("realsim_10000_600_meka.txt", dataX, fmt="%0.8f")
    np.savetxt("realsim_10000_label.txt", y, fmt="%0.1f")

main()