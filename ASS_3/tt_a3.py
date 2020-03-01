'''
Name : Kothapalli Sandeep
roll : 17CS10021
pandas and numpy have been used.
random has been used to generate random sampling with bias towards datapoints with higher weight
'''
import numpy as np  
import pandas as pd
import math
import random
from pprint import pprint
X_data = pd.read_csv('data3_19.csv')
test_data = pd.read_csv('test3_19.csv',header= None)
header = list(X_data.columns.values)

def entropy (Y):
    ent = 0
    values = Y.unique()
    for val in values :
        frac = Y.value_counts()[val] / len(Y)
        ent = ent - frac*math.log(frac)
    return ent
def gain (X,Y,attribute):
    entX = entropy(Y)
    values = X[attribute].unique()
    ent_sum = 0
    for val in values :
        index = X.index[X[attribute]==val].tolist()
        Y_temp = Y.iloc[index].reset_index(drop=True)
        frac = len(Y_temp)/len(Y)
        ent_sum = ent_sum + frac*entropy(Y_temp)
    _gain = (entX-ent_sum)
    return _gain
def decideAttribute (X,Y):
    attribute = X.keys()[0]
    _gain = 0
    for att in X.keys() :
        temp = gain (X,Y,att)
        if (temp > _gain) :
            _gain = temp
            attribute = att
    return attribute
def getSub (X,Y,att,val):
    index = X.index[X[att]==val].tolist()
    X_temp = X.iloc[index,:].reset_index(drop=True)
    Y_temp = Y.iloc[index].reset_index(drop=True)
    return X_temp,Y_temp

def buildTree(X,Y,count,p_att,tree=None):
    att = decideAttribute(X,Y)
    values = X[att].unique()
    if tree is None :
        tree = {}
        tree[att] = {}
    for val in values :
        XsubTable,YsubTable = getSub (X,Y,att,val)
        y_values = YsubTable.unique()
        yes = 0
        no = 0
        for  y_val in y_values :
            if(y_val=='yes'):
                yes = YsubTable.value_counts()['yes']
            if (y_val=='no'):
                no = YsubTable.value_counts()['no']
        if(att == p_att):
            if(yes > no):
                return 'yes'
            else :
                return 'no'
        elif (count > 1) :
            if(yes > no):
                tree[att][val] = 'yes'
            else :
                tree[att][val] = 'no'
        else :
            tree[att][val] = buildTree(XsubTable,YsubTable,count+1,att)
    return tree



def predict_recur(x, _next):
    if not isinstance(_next, dict):
        return _next
    pos = header.index(list(_next.keys())[0])
    x_value = x[pos]
    return predict_recur(x, _next[list(_next.keys())[0]][x_value])

def predicto(x,tree):
    return predict_recur(x, tree)

weight = [0] * X_data.shape[0]
header.append("weight")
X_data.insert(4,"weight",[1/X_data.shape[0]] * X_data.shape[0],True)

class AdaBoostClassifier:
    def __init__(self, training_data, header, rounds,global_data):
        self.training_data = training_data.values.tolist()
        self.n = len(training_data)
        self.header = header
        self.rounds = rounds
        self.alphas = []  
        self.trees = []
        self.global_data = global_data.values.tolist()

    def encoder(self, pred):
        if (pred.upper() == 'YES'):
            return 1
        else:
            return -1

    def build(self):
        for i in range(self.rounds):
            data = pd.DataFrame(self.training_data, columns=self.header)
            X_train = data.iloc[:,0:3].reset_index(drop=True)
            Y_train = data.iloc[:, 3].reset_index(drop=True)
            tree =  buildTree(X_train,Y_train,0,None)
            predictions = [predicto(self.training_data[j][0:3],tree) for j in range(self.n)]
            wrong = 0
            for j in range(self.n):
                if self.encoder(predictions[j]) != self.encoder(self.training_data[j][-2]):
                    wrong += 1
            # uncomment to see error rate  and count
            # print(f"WRONG: {wrong} and TOTAL: {self.n}")
            error_rate = [(self.encoder(predictions[j]) != self.encoder(self.training_data[j][-2]))* self.training_data[j][-1] for j in range(self.n)]
            error_rate = sum(error_rate)
            # error_rate  /= self.n
            # print(f"Iteration {i} :")
            # print(f"\tError rate for iteration {i} is {error_rate}.")
            # print(error_rate)
            # appending since empty list initially 
            self.alphas.append(0.5*np.math.log((1-error_rate)/error_rate)) 
            # print(f"\tCalculated alpha for tree {i} is {self.alphas[i]}.")
            # weight_sum = sum([self.training_data[j][-1] for j in range(self.n)]) #zi old  w
            for j in range(self.n):
                hx = self.encoder(predictions[j])
                yx = self.encoder(self.training_data[j][-2])
                # incr if wrong and decr if correct
                if(hx == yx):
                    self.training_data[j][-1] = \
                        (self.training_data[j][-1])*math.exp(-1*self.alphas[i])
                else:
                    self.training_data[j][-1] = \
                        (self.training_data[j][-1])*math.exp(self.alphas[i])
                # commented the algo found in the  research paper 
                # self.training_data[j][-1] = \
                # (self.training_data[j][-1]/weight_sum)*math.exp(-1*self.alphas[i]*hx*yx)
        # normalising the training data weights - not required bc normalising later
            # weight_sum = sum([self.training_data[j][-1] for j in range(self.n)])
            # for  j  in range (self.n) :
            #     self.training_data[j][-1] /= weight_sum
            # update the original data corresponding to the weights



            # can be optimised by selecting unique in current classifier set but time nhi h    
            # updating weights in our global data
            for i in range (self.n):
                curr = self.training_data[i][0:-1]
                for j in range (self.n):
                    if(self.global_data[j][0:-1] == curr) :
                        self.global_data[j][-1] = self.training_data[i][-1]
            # normalise the new weights in the original data set

            normalize_factor = sum ([self.global_data[j][-1] for j in range (self.n)])

            for j in range (self.n):
                self.global_data[j][-1] /= normalize_factor
# choose randomly from original data set with correspondingly updated weights
# accuracy might show  a little fluctuation due to random sampling of the set but it doesnot fall below the 
# accuracy of our original decision tree , working as expected
            new_data = random.choices(
                population = self.global_data,
                weights = [self.global_data[s][-1] for s in range(self.n)],
                k = self.n
            )
            self.training_data = new_data
            self.trees.append(tree)
            # print("\tTree is :")
            # pprint(tree)
        # if necessary uncomment to see the  tree in each iteration of the adaboost algorithm
        # print(self.alphas)
        # pprint(self.trees)
        
    def predict(self, x):
        h_sum = 0
        for i in range(self.rounds):
            h_sum += self.alphas[i] * self.encoder(predicto(x,self.trees[i]))
        if (h_sum > 0):
            return "YES"
        else:
            return "NO"

test_data = test_data.values.tolist()

#multiple lines in the build adaboost have been commented out
# uncommenting them will display the error rate and count in each iteration
# specificities have been mentioned wherever necessary
print("Adaboost Classifier has started training.\n")
ada_classifier = AdaBoostClassifier(X_data, header, 3,X_data)
ada_classifier.build()
print("Training Complete !\n")
corr_count = 0
# print("#############################################################")
print("Testing the trained classifier on test3_19.csv")
for i in range(len(test_data)):
    verdict = "WRONG"
    pred = ada_classifier.predict(test_data[i][0:3])
    if pred == test_data[i][-1].upper():
        verdict = "CORRECT"
        corr_count += 1
    # uncomment to see the verdict if each test case
    # print(f"{i}. {test_data[i]} - Prediction = {pred} :: Verdict = {verdict} prediction")
# print("#############################################################")
# uncomment to see trees
# pprint(ada_classifier.trees)
# getting equal accuracy because alpha of initial tree dominates
print(f"{corr_count} out of {len(test_data)} correct predictions.")
print(f"ACCURACY: {100*corr_count/len(test_data)}")
# since random.choices gives a fairly unique sample in every iteration,
# our results would be more confident if we take an average of accuracies over a considerably large number of
# iterations. uncomment the  below lines of code  for  a  better estimate of the accuracy of the algorithm. 
# average over 1000 iterations
# accuracyList = []
# for j in range(2):
#     ada_classifier = AdaBoostClassifier(X_data, header, 3,X_data)
#     ada_classifier.build()
#     corr_count = 0
#     # pprint(X_data)
#     for i in range(len(test_data)):
#         pred = ada_classifier.predict(test_data[i][0:3])
#         if pred == test_data[i][-1].upper():
#             corr_count += 1
#     accuracy = 100*corr_count/len(test_data)
#     # print(f"ACCURACY: {accuracy}")
#     accuracyList.append(accuracy)
#     print("bobs")

# print(sum(accuracyList)/len(accuracyList))



