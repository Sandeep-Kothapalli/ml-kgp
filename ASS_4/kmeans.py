import random
import sys
import math
import pandas as pd
# from math import *
from pprint import pprint
def ReadData(fileName):
    f = open(fileName, 'r')
    lines = f.read().splitlines()
    f.close()
    items = []
    for i in range(0, len(lines)-1):
        line = lines[i].split(',')
        itemFeatures = []
        for j in range(len(line)-1):
            v = float(line[j])
            itemFeatures.append(v) 
        items.append(itemFeatures)
        # random.shuffle(items)
    return items


# def FindColMinMax(items):
#     n = len(items[0])
#     minima = [sys.maxsize for i in range (n)]
#     maxima = [-sys.maxsize -1 for i in range(n)]

#     for item in items :
#         for f in range(len(item)):
#             if(item[f] < minima[f]):
#                 minima[f] = item[f]
#             if(item[f] > maxima[f]):
#                 maxima[f] = item[f]
#     return minima,maxima

def InitializeMeans (items,k):
    f = len(items[0])
    means = [[0 for i in range(f)] for j in range(k)]
    for i in range(len(means)):
        means[i] = items[random.randint(0,len(items)-1)]
    # pprint(means)
    return means

def EuclideanDistance(x,y):
    s = 0 
    for i in range(len(x)):
        s += math.pow(x[i]-y[i],2)
    return math.sqrt(s)

def UpdateMean (n,mean,item):
    # print(item)
    # print(mean)
    for i in range(len(mean)):
        m = mean[i]
        m = (m*(n-1)+item[i])/float(n)
        mean[i] = round(m, 3)
    return mean

def Classify(means,item):
    minimum = sys.maxsize
    index = -1
    for i in range(len(means)): 
        dis = EuclideanDistance(item, means[i])
        if (dis < minimum): 
            minimum = dis
            index = i
    return index

def CalculateMeans(k,items,maxIterations=10): 
	# cMin, cMax = FindColMinMax(items) 
	means = InitializeMeans(items,k) 
	clusterSizes= [0 for i in range(len(means))] 
	belongsTo = [0 for i in range(len(items))] 
	for e in range(maxIterations): 
		noChange = True 
		for i in range(len(items)): 
			item = items[i]  
			index = Classify(means,item) 
			clusterSizes[index] += 1 
			cSize = clusterSizes[index] 
			means[index] = UpdateMean(cSize,means[index],item) 
			if(index != belongsTo[i]): 
				noChange = False 
			belongsTo[i] = index 
		if (noChange): 
			break 
	return means

def FindClusters(means,items): 
	clusters = [[] for i in range(len(means))]
	for item in items: 
		index = Classify(means,item) 
		clusters[index].append(item)
	return clusters

def jaccard_similarity(list1, list2):
    list1 = [str(row) for  row in list1]
    list2 = [str(row) for  row in list2]
    intersection = len(set(list1).intersection(list2))
    union = (len(list1) + len(list2)) - intersection
    return  (intersection / union)

def main():
    fileName = "data4_19.csv"
    items = ReadData(fileName)
    # pprint(items)
    # print(len(items))
    X = pd.read_csv(fileName,header=None)
    # print(X[4])
    mismatch = []
    for i in range (len(X[4])-1):
        if(X[4][i] != X[4][i+1]):
            # print(f"mismatch at index {i}")
            mismatch.append(i)
    flowerList = []
    flowerList.append(items[0:mismatch[0]+1])
    flowerList.append(items[mismatch[0]+2:mismatch[1]+1])
    flowerList.append(items[mismatch[1]+2:len(X)])
    # pprint(flowerList)
    random.shuffle(items)
    k = 3
    means = CalculateMeans(k,items)
    clusters = FindClusters(means,items)
    # maxLen = -sys.maxsize
    # uncomment for hack for better  output
    # maxLen = max(len(clusters[0]),len(clusters[1]),len(clusters[2]))
    # while(maxLen >= 70):
    #     clusters = FindClusters(means,items)
    #     maxLen = max(len(clusters[0]),len(clusters[1]),len(clusters[2]))


    # print(len(means))
    # pprint (means)
    
    # find jaccard distance and classify the clusters
    # find means of each iris  and calculate jaccard for each combination
    # select 0 to mismatch[0]
    # mismatch[0]+1 to mismatch[1]
    # mismatch[1] to end
    # print(type(items))
    # pprint(flowerList)
    # pprint(flowerList[0])
    flowerNames = ["Iris-setosa","Iris-versicolor","Iris-virginica"]

    # print(flowerNames[0])
    # jaccard_similarity(list1,list2)
    # print(type(clusters[0]))
    
    for i in range(len(clusters)):
        print(f"Cluster {i+1} : \n\tMean =  {means[i]}")
        # pprint(clusters[i])
        print(f"\tSize of Cluster {i+1} = {len(clusters[i])}")
        a = jaccard_similarity(flowerList[0],clusters[i])
        b = jaccard_similarity(flowerList[1],clusters[i])
        c = jaccard_similarity(flowerList[2],clusters[i])
        if(a >= b):
            if(b >= c):
                print(f"\tCluster best represents {flowerNames[0]}")
                print(f"\tJaccard-distance with {flowerNames[0]} is {round(1-a,4)}")
            elif(c >= a):
                print(f"\tCluster best represents {flowerNames[2]}")
                print(f"\tJaccard-distance with {flowerNames[2]} is {round(1-c,4)}")
        else : #b > a
            if(c >= b):
                print(f"\tCluster best represents {flowerNames[2]}")
                print(f"\tJaccard-distance with {flowerNames[2]} is {round(1-c,4)}")
            else :
                print(f"\tCluster best represents {flowerNames[1]}")
                print(f"\tJaccard-distance with {flowerNames[1]} is {round(1-b,4)}")
        # print(f"\tJaccard-distance with {flowerNames[0]} = {a}")
        # print(f"\tJaccard-distance with {flowerNames[1]} = {b}")
        # print(f"\tJaccard-distance with {flowerNames[2]} = {c}")
        

if __name__ == "__main__":
    main()
