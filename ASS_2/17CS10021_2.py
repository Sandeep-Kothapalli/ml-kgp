# Kothapalli Sandeep
# 17CS10021
# assignment 2
#please ignore pylint(unsubscriptable-object) error
#the code works perfectly. and it does not modify either of the data or test files
#no compilation flags required
import sys
import pandas as pd
from pprint import pprint 
class nbc :
    def __init__ (self,x,y) :
        self.x = x
        self.y = y
        self.Xtable = None
        self.Dtable = None
        self.classes = y.unique()  
    def buildXtable(self) :
        #building frequency and probability tables
        self.Xtable = {}
        for col in self.x.columns :
            #creating table for each column
            self.Xtable[col] = {}
            uniqueRank = self.x[col].unique()
            #for each unique rank calculate probabilities
            for rank in uniqueRank :
                self.Xtable[col][rank] = {}
                #subset is  the set of indices of items with value = rank
                subset = self.x.index[self.x[col] == rank].tolist()
                #print(rank)
                #print(subset)
                for truthValue in self.classes :
                    count = 0
                    for idx in subset :
                        if self.y[idx] == truthValue :
                            count += 1
                    #implementing below line gives a higher accuracy and is same as the value when tested with
                    #multinomialMB from sklearn : clarify with teacher about the difference
                    #self.Xtable[col][rank][truthValue] = (count+1)/(6+self.x[col].value_counts()[rank])
                    #reference course slides
                    self.Xtable[col][rank][truthValue] = (count+1)/(5+self.Dtable[truthValue])   
                    #adding 1 to numerator  and 5(number of classes a attribute can take) to  denominator are part of  laplacian smoothing
                    #reference course slides   
        #pprint(self.Xtable)
        #pprint(self.Dtable)
        return
    def buildDtable(self) :
        self.Dtable = {}
        for truthValue in self.classes :
            self.Dtable[truthValue] = self.y.value_counts()[truthValue]#
            self.x.shape[0]
            #probability of truth and probability of false
            #the  number of instance of  each  truth value in class is stored for usability in the buildtable method
            #uncomment self.x.shape[0] to update the values to probability  and  make necessary changes in the build table  function
    def trainClassifier (self) :
        print("Training started.\n")
        self.buildDtable()
        self.buildXtable()
        print("Training complete! The Probability tables are generated(with Laplacian Smoothing).\n")
    def predictProbability(self,testX) :
        maxProb = -1
        maxProbBool = -1
        for  truthValue in self.classes :
            prod = 1
            for col in  self.x.columns :
                col_idx = self.x.columns.get_loc(col)
                prod *= self.Xtable[col][testX[col_idx]][truthValue]
            prod*= self.Dtable[truthValue]
            if prod > maxProb :
                maxProb = prod
                maxProbBool = truthValue
        return maxProbBool

def getData (fileName) :
    array = []
    content = None
    with open(fileName) as f :
        content = f.readlines()
    for row in content :
        if row[0] == '"' :
            array.append(row[1:-2].replace("\n","").replace(" ","").split(","))
        else :
            array.append(row.replace("\n","").replace(" ","").split(",")) 
    return pd.DataFrame(array[1:],columns = array[0])
        
def main() :
    df = getData("data2_19.csv")
    #print(df)    
    #print(df.columns[0])
    trainX = df[df.columns[1:]]
    trainD = df[df.columns[0]]
    #print (trainX)
    #print (trainD)
    classifier = nbc(trainX,trainD)
    classifier.trainClassifier()
    df = getData("test2_19.csv")
    testX = df[df.columns[1:]]
    testD = df[df.columns[0]]
    accuracy = 0
    i = 1 
    print("Testing started  !!\n")
    #value = False
    print("############################################################################################")
    for index,row in testX.iterrows() :
        value = False
        if classifier.predictProbability(row) == testD[index] :
            accuracy += 1 
            value = True
        if  value :
            print(f"For test set instance {i},\tPrediction : [{classifier.predictProbability(row)}], Actual : [{testD[index]}]\t\tCORRECT PREDICTION")
        else :
            print(f"For test set instance {i},\tPrediction : [{classifier.predictProbability(row)}], Actual : [{testD[index]}]\t\tWRONG PREDICTION")                     
        i+=1 
    print("############################################################################################")
    print(f"{accuracy} out of {len(testD)} values are predicted correctly.")
    print(f"\n\nAccuracy : {accuracy/len(testD)}")

if __name__ == '__main__' : 
    main()