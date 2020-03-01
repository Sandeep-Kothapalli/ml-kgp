"""
Kothapalli Sandeep
17CS10021
Assignment4 : K means clustering
"""
import random 
import time
#for ramdom choices and reshuffling of data
import sys
import numpy as np
import pandas as pd
from pprint import pprint
# initialising means as k random items from original data
def initMeans(k,items):
    means = [[0 for i in range(len(items[0]))] for j in range(k)]
    means = random.choices(items,k = len(means))
    # not using the below because there maybe a  chance that a mean is repeated.
    # for i in range(len(means)):
    #     # selects a random data point considering a uniform distribution
    #     means[i] = random.choice(items)
    return means

# updating the mean for every iteration / modification of a point
def newMean(items, w):
    centroid = [0] * w
    for i in range(len(items)):
        for j in range(w):
            centroid[j]  += items[i][j]
    for j in range(w):
        centroid[j] /= len(items)
    return centroid

# classifying the point, which cluster does it belong to
def whichCluster(k,means,item):
    min_dist = sys.maxsize
    b = np.array(item)
    for x in range(k):
        a = np.array(means[x])
        # numpy function to calculate the euclidean distance
        # check documentation https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
        dist = np.linalg.norm(a-b)
        if(dist < min_dist):
            min_dist = dist
            idx = x
    return idx

# finding k means, max iterations= 10
def findMeans(k,items,iterations = 10):
    means = initMeans(k,items)
    clusterSizes= [0 for i in range(len(means))]
    belongsTo = [0 for i in range(len(items))]
    # self explanatory code
    # finding clusters and updating the  means.
    for e in range(iterations):
        flag = True
        for i in range(len(items)):
            item = items[i]  
            index = whichCluster(k,means,item) 
            clusterSizes[index] += 1 
            if(index != belongsTo[i]): 
                flag = False 
            belongsTo[i] = index
            if (flag):
                break
        clusters = findClusters(k,means,items)
        # update mean
        for i in range(k):
            means[i] = newMean(clusters[i], len(means[i]))
    return means,clusters

# finding the clusters
def findClusters(k,means,items):
    clusters = [[] for i in range(k)]
    for i in items :
        idx = whichCluster(k,means,i)
        clusters[idx].append(i)
    return clusters

# jaccard similarity
def jaccard_similarity(list1, list2):
    list1 = set([str(row) for  row in list1])
    list2 = set([str(row) for  row in list2])
    intersection = len(list1 & list2)
    union = len(list1 | list2)
    return  (intersection / union)

def main():
    random.seed(time.time())
    # please run the code multiple times for better accuracy testing
    fileName = "data4_19.csv"
    data = pd.read_csv(fileName,header=None)
    trainX = data.iloc[:,:-1]
    # used for training
    items = trainX.values.tolist()
    mismatch = []
    for i in range (len(data[4])-1):
        if(data[4][i] != data[4][i+1]):
            mismatch.append(i)
            # flowerList is the original data set categorized by the type of iris
            # this is only  used for finding jaccard distance , not for training
    flowerList = []
    flowerList.append(items[0:mismatch[0]+1])
    flowerList.append(items[mismatch[0]+2:mismatch[1]+1])
    flowerList.append(items[mismatch[1]+2:len(data)])   
    random.shuffle(items)
    k = 3
    # finding k means, wtih specified number of iterations
    means,clusters = findMeans(k,items)
    flowerNames = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
    for i in range(len(clusters)):
            for j in range(len(means[i])):
                means[i][j] = round(means[i][j],4)
            print(f"Cluster {i+1} : \n\tMean =  {means[i]}")
            print(f"\tSize of Cluster {i+1} = {len(clusters[i])}")
            # finding the cluster a  certain point belongs to with jaccard similarity measure. 
            a = jaccard_similarity(flowerList[0],clusters[i])
            b = jaccard_similarity(flowerList[1],clusters[i])
            c = jaccard_similarity(flowerList[2],clusters[i])
            # printing iris classes for each calculated cluster.
            if(a >= b):
                if(b >= c):
                    print(f"\tCluster best represents {flowerNames[0]}")
                    print(f"\tJaccard-distance with {flowerNames[0]} is {round(1-a,4)}")
                elif(c >= a):
                    print(f"\tCluster best represents {flowerNames[2]}")
                    print(f"\tJaccard-distance with {flowerNames[2]} is {round(1-c,4)}")
            else : 
                if(c >= b):
                    print(f"\tCluster best represents {flowerNames[2]}")
                    print(f"\tJaccard-distance with {flowerNames[2]} is {round(1-c,4)}")
                else :
                    print(f"\tCluster best represents {flowerNames[1]}")
                    print(f"\tJaccard-distance with {flowerNames[1]} is {round(1-b,4)}")        

if __name__ == "__main__":
    main()