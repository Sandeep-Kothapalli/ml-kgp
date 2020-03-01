#Kothapalli Sandeep
#17CS10021
#assignment 1
#no compilation flags rewuired
import numpy as np
import math
import pandas as pd
from numpy import log2 as log
# small value is to avoid the divide by zero error runtime warning
smallvalue = np.finfo(float).eps
X = pd.read_csv('data1_19.csv')
def entropy(data):
    yes = 0
    no = 0
    for row in data:
        if row[-1] == 'yes':
            yes += 1
        else:
            no += 1
    if yes == 0 or no == 0:
        return 0
    entropy = -(yes/(yes+no))*(math.log(yes/(yes+no), 2)) -(no/(yes+no))*(math.log(no/(yes+no), 2))
    return entropy

# Function to calculate entropy attribute
def entropy_att(data,attribute):
  Class = data.keys()[-1] 
  target_variables = data[Class].unique()  
  variables = data[attribute].unique()    
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(data[attribute][data[attribute]==variable][data[Class] ==target_variable])
          den = len(data[attribute][data[attribute]==variable])
          fraction = num/(den+smallvalue)
          entropy += -fraction*log(fraction+smallvalue)
      fraction2 = den/len(data)
      entropy2 += -fraction2*entropy
  return abs(entropy2)

# Finding the decider node this can also be done in two different functions. one to calculate the info gain and other to find best split
def bestSplitNode(dataSegment):
    info_gain = []
    for key in dataSegment.keys()[:-1]:#[:-1] is to exclude the last col which is result
        info_gain.append(entropy(dataSegment)-entropy_att(dataSegment,key))
    return dataSegment.keys()[:-1][np.argmax(info_gain)]

def checkSubset(data,node,value):
    return (data[node] == value)
    
def get_subset(dataSegment, node,value):
    flag = checkSubset(dataSegment,node,value)
    return dataSegment[flag].reset_index(drop=True)

# Build Decision Tree
def buildTree(dataSegment,count,maxlevels,tree=None): 
    node = bestSplitNode(dataSegment)
    attValue = np.unique(dataSegment[node])    
    if tree is None:                    
        tree={}
        tree[node] = {} #node = header[bestCol]
    for value in attValue:        
        subtable = get_subset(dataSegment,node,value)
        clValue,counts = np.unique(subtable['survived'],return_counts=True)                         
        if len(counts)==1 or count==maxlevels:
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable,count+1,maxlevels)
                   
    return tree    
def print_tree(dic,level):
    #print("modify source code to see the tree at different depths")
    if tree is None:
        return
    if type(dic)!=dict:
        print(": "+dic)
        return
    for key in dic:
        print()
        val = dic[key]
        if type(val)==dict:
            for k in val:
                for i in range(level):
                    print("\t",end="")
                print("|"+key+" = "+str(k),end=" ")
                print_tree(val[k],level+1)

maxDepth = 2 
tree = buildTree(X,0,maxDepth)
print_tree(tree,0)

#predict (X,tree)