import numpy as np
import random
import math
from scipy.linalg import svd
from sklearn.metrics.pairwise import euclidean_distances
MAXDIS = 1000000000000000.0
tol = 0.0000000001

def KMeans(X, clusterNum, maxIter, colNum, taken):
    totalWeight = taken
    idx = np.full((taken),-1)
    #centers = np.zeros((clusterNum,colNum))
    xnorm = np.zeros((taken))
    for i in range(taken):
        for j in range(colNum):
            xnorm[i] += X[i,j]*X[i,j]

    for i in range(taken):
        idx[i] = random.randint(0,clusterNum-1)

    for iter in range(maxIter):
        centers = np.zeros((clusterNum,colNum))
        count = np.zeros((clusterNum))
        for i in range(taken):
            for j in range(colNum):
                centers[idx[i],j] += X[i,j]
            count[idx[i]] += 1

        for i in range(clusterNum):
            for j in range(colNum):
                centers[i,j] /= count[i]

        centernorm = np.zeros((clusterNum))
        for i in range(clusterNum):
            for j in range(colNum):
                centernorm[i] += centers[i,j]*centers[i,j]

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

def NysTrain(Y, center, m, ki, gamma):
    #n = len(Y)
    #m = int(m)
    #randPerm = np.random.permutation(n)
    #randPerm = list(range(n))
    #takenIndex = randPerm[:m]
    #center = Y[np.ix_(takenIndex)]

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
    filename = "ijcnn1/ijcnn1"
    clusterNum = 4
    maxIter = 10
    dataNum = 49990
    colNum = 22
    taken = 48000
    targetRank = 128
    gamma = 0.125
    eta = 0.1
    partData = int(taken/clusterNum)

    file1 = open(filename, "r+")
    count = 0
    X = np.zeros((taken, colNum))
    y = np.zeros((taken, 1))
    randPerm = list(range(dataNum))
    #randPerm = np.random.permutation(dataNum)
    takenIndex = randPerm[:taken]

    for i in range(taken):
        dataLine = file1.readline()
        #if i in takenIndex:
        dataLine = dataLine.strip()
        splitBySpace = dataLine.split(' ')
        y[count, 0] = float(splitBySpace[0])
        dataLen = len(splitBySpace)
        for j in range(dataLen - 1):
            temp = splitBySpace[j + 1]
            splitByColon = temp.split(':')
            X[count, int(splitByColon[0]) - 1] = float(splitByColon[1])
        count += 1
    print("Input completes")

    idx, centers = KMeans(X,clusterNum,maxIter,colNum,taken)
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

    #new part
    geolst = []
    for i in range(1):
        templst = []
        for k in range(clusterNum):
            indices = indexes[k]
            indx = [x for j, x in enumerate(indices) if x >= partData*i and x<partData*(i+1)]
            templst.append(indx)
            print(indx)
        geolst.append(templst)
    geoIndexes = np.array(geolst)
    #new part ends

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

    #new part
    sharedlst = []
    sharedA = []
    for i in range(clusterNum):
        templst = []
        tempA = []
        for k in range(clusterNum):
            randPermi = np.random.permutation(len(geolst[i][k]))
            numi = min(4 * int(ranklist[k]), len(geolst[i][k]))
            randi = randPermi[:numi]
            res_list_i = [geolst[i][k][p] for p in randi]
            res_A = X[np.ix_(res_list_i)]
            templst.append(res_list_i)
            tempA.append(res_A)
        sharedlst.append(templst)
        sharedA.append(tempA)

    print("K means complete")

    #S = np.identity(targetRank, dtype=float)
    Ulist = []
    for k in range(clusterNum):
        ki = min(ranklist[k], lengths[k])
        ranklist[k] = ki
        m = min(2*ki, lengths[k])
        Y = X[np.ix_(indexes[k])]
        totalSample = []
        for i in range(clusterNum):
            totalSample.extend(sharedlst[i][k])
        random.shuffle(totalSample)
        B = X[np.ix_(totalSample[:int(m)])]
        U = NysTrain(Y, B, m, ki, gamma)
        Ulist.append(U)

    print("U list complete")

    #Slist = []
    #for s in range(1):
    S = np.identity(targetRank, dtype=float)
    #index = geoIndexes[s]
    for i in range(clusterNum):
        for j in range(i+1,clusterNum):
            if dis[i,j] >= threshold:
                totalSample_i = []
                totalA_i = []
                for k in range(clusterNum):
                    totalSample_i.extend(sharedlst[k][i])
                    totalA_i.extend(sharedA[k][i])
                randi=[]
                for k in range(len(indexes[i])):
                    if indexes[i][k] in totalSample_i:
                        randi.append(k)

                totalSample_j = []
                totalA_j = []
                for k in range(clusterNum):
                    totalSample_j.extend(sharedlst[k][j])
                    totalA_j.extend(sharedA[k][j])
                randj = []
                for k in range(len(indexes[j])):
                    if indexes[j][k] in totalSample_j:
                        randj.append(k)
                Ai = np.array(totalA_i)
                Aj = np.array(totalA_j)
                #randPermi = list(range(int(lengths[i])))
                #randPermi = np.random.permutation(int(lengths[i]))
                #numi = min(4*int(ranklist[i]),int(lengths[i]))
                #randi = randPermi[:numi]
                #randPermj = list(range(int(lengths[j])))
                #randPermj = np.random.permutation(int(lengths[j]))
                #numj = min(4 * int(ranklist[j]), int(lengths[j]))
                #randj = randPermj[:numj]
                Ui = Ulist[i][np.ix_(randi)]
                Uj = Ulist[j][np.ix_(randj)]
                #res_list_i = [indexes[i][k] for k in randi]
                #Ai = X[np.ix_(res_list_i)]
                #res_list_j = [indexes[j][k] for k in randj]
                #Aj = X[np.ix_(res_list_j)]
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

                '''randPermi = np.random.permutation(len(index[i]))
                numi = min(4 * int(ranklist[i]), len(index[i]))
                randi = randPermi[:numi]
                randPermj = np.random.permutation(len(index[j]))
                numj = min(4 * int(ranklist[j]), len(index[j]))
                randj = randPermj[:numj]
                res_list_i = [index[i][k] for k in randi]
                Ai = X[np.ix_(res_list_i)]
                res_list_j = [index[j][k] for k in randj]
                Aj = X[np.ix_(res_list_j)]
                res_list_ui = []
                for k in range(int(lengths[i])):
                    if indexes[i][k] in res_list_i:
                        res_list_ui.append(k)
                res_list_uj = []
                for k in range(int(lengths[j])):
                    if indexes[j][k] in res_list_j:
                        res_list_uj.append(k)
                Ui = Ulist[i][np.ix_(res_list_ui)]
                Uj = Ulist[j][np.ix_(res_list_uj)]
                tmpK = squareDist(Ai, Aj, gamma=gamma)
                Z1 = np.dot(np.linalg.pinv(np.dot(Ui.T, Ui), rcond=1e-6), Ui.T)
                Z2 = np.dot(Uj, np.linalg.pinv(np.dot(Uj.T, Uj), rcond=1e-6))
                Z = np.dot(Z1, np.dot(tmpK, Z2))
                si = int(sumRankList[i])
                sj = int(sumRankList[j])
                r = len(Z)
                c = len(Z[0])
                S[si:si + r, sj:sj + c] = Z
                S[sj:sj + c, si:si + r] = Z.T
    Slist.append(S)'''


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
    '''full_approx = np.zeros((testNum,taken))
    r=0
    for k in range(clusterNum):
        demo_index = []
        for i in testIndex:
            if i>=k*partData and i<(k+1)*partData:
                demo_index.append(i)
        U_samp = finalU[np.ix_(demo_index)]
        partial_approx = np.dot(U_samp,np.dot(Slist[0],finalU.T))
        full_approx[r:r+len(U_samp),:] = partial_approx'''
    U_samp = finalU[np.ix_(testIndex)]
    err = np.linalg.norm(tmpK-(np.dot(U_samp,np.dot(S,finalU.T))))/np.linalg.norm(tmpK)
    #err = np.linalg.norm(tmpK-full_approx)/np.linalg.norm(tmpK)
    print("Error is "+str(err))

main()