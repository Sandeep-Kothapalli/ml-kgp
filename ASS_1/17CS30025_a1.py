# Roll No : 17CS30025
# Name : Pericherla Amshumaan Varma
# Assignment No : 1
# Pandas, Numpy and Math lib have been used

# pandas library has been used just as a datatype for input data. No functions in pandas that build the tree directly have been used

# import libraries
import numpy as np
import pandas as pd
import math


# find entropy for current data without any split
def entropy(Y):  

    Ent = 0
    values = Y.unique()
    for val in values:
        frac = Y.value_counts()[val] / len(Y)
        Ent = Ent - frac * math.log(frac)
    return Ent



# calculate entropy gain on splitting with att attribute
def gain(X, Y, att):

	Ent_X = entropy(Y)
	values = X[att].unique()
	Ent_sum = 0
	for val in values:
		index = X.index[X[att] == val].tolist()
		Y_temp = Y.iloc[index]
		Y_temp = Y_temp.reset_index(drop=True)
		frac = len(Y_temp)/len(Y)
		Ent_sum = Ent_sum + frac * entropy(Y_temp)
	return (Ent_X - Ent_sum)



# Decide on which attribute to split based on Highest gain
def decide_att(X, Y):

	attribute = X.keys()[0]
	_gain = 0
	for att in X.keys():
		temp = gain(X, Y, att)
		if temp > _gain:
			_gain = temp
			attribute = att
	return attribute



# Get the sub data after splitting wit att attribute
def get_sub_data(X, Y, att, val):

	index = X.index[X[att] == val].tolist()
	X_temp = X.iloc[index, : ]
	Y_temp = Y.iloc[index]
	X_temp = X_temp.reset_index(drop=True)
	Y_temp = Y_temp.reset_index(drop=True)
	return X_temp, Y_temp



# Get the decision Tree based on Information gain, withou any pruning at any level
def get_tree(X, Y, count, p_att, tree=None):
	att = decide_att(X, Y)
	values = X[att].unique()
	if tree is None:                    
		tree = {}
		tree[att] = {}
	
	for val in values:
		X_sub, Y_sub = get_sub_data(X, Y, att, val)
		y_values = Y_sub.unique()
		yes = 0
		no = 0
		for y_val in y_values:
			if y_val == 'yes':
				yes = Y_sub.value_counts()['yes']
			if y_val == 'no':
				no = Y_sub.value_counts()['no']
		
		if no == 0:
			ratio = 150
		else :
			ratio = float(yes / no)

		if att == p_att:
			if yes > no:
				return 'yes'
			else:
				return 'no'
# Uncomment this part to stop growing if any one is much greater than other, Pruning
#		elif ratio > 10 :
#			tree[att][val] = 'yes'
#		
#		elif ratio < 0.1 :
#			tree[att][val] = 'no'

		elif count > 1: # As only 3 attributes in this case
			if yes > no:
				tree[att][val] = 'yes'
			else:
				tree[att][val] = 'no'
		
		else :
			tree[att][val] = get_tree(X_sub, Y_sub, count+1, att)
		
	return tree


# To print the decision tree as per given notation
def print_tree(dic,level):
    if type(dic)!=dict:
        print(": "+dic)
        return
    for key in dic:
        print()
        val = dic[key]
        if type(val)==dict:
            for k in val:
            	for i in range(level-1):
                    print("\t",end="")
            	if level!=0 :
            		print("|---->", end="")
            		print("\t",end="")
            	print(key+" = "+str(k),end=" ")
            	print_tree(val[k],level+1)


# Function to detect no of correct and incorrect predictions
def test_accuracy(X_test, Y_test, tree):
	correct = 0
	incorrect = 0
	for index, row, in X_test.iterrows():
	    ini = tree.keys()
	    i = list(ini)[0]
	    val1 = row[i]
	    sec = tree[i].keys()

	    for j in sec:
	    	
	    	if val1 == j:
	    		
	    		thr = tree[i][j]
	    		if (thr == 'yes') | (thr == 'no'):
	    			if Y_test[index] == thr:
	    				correct += 1
	    				continue
	    			else:
	    				incorrect += 1
	    				continue
	    			
	    		thr = thr.keys()
	    		k = list(thr)[0]

	    		val2 = row[k]

	    		fou = tree[i][j][k].keys()
	    		for l in fou:
	    			if val2 == l:
	    				fiv = tree[i][j][k][l]
	    				if (fiv == 'yes') | (fiv == 'no'):
	    					if Y_test[index] == fiv:
	    						correct += 1
	    						continue
	    					else:
	    						incorrect += 1
	    						continue

	    					
	    				fiv = fiv.keys()
	    				m = list(fiv)[0]

	    				val3 = row[m]
	    				
	    				six = tree[i][j][k][l][m].keys()
	    				for n in six:
	    					if val3 == n:
		    					sev = tree[i][j][k][l][m][n]
		    					if Y_test[index] == sev:
		    						correct += 1
		    						continue
		    					else:
		    						incorrect += 1
		    						continue
	return correct, incorrect
             


# Input Data From the given file
X_data = pd.read_csv('data1_19.csv')


# Uncomment this section and comment the below one to Split the data into test and training sets 
'''
shuffled_indices=np.random.permutation(len(X_data))
test_set_size= int(len(X_data)*0.2)
test_indices=shuffled_indices[:test_set_size]
train_indices=shuffled_indices[test_set_size:]
X_train = X_data.iloc[train_indices,:-1].reset_index(drop=True)
X_test = X_data.iloc[test_indices,:-1].reset_index(drop=True)
Y_train = X_data.iloc[train_indices,-1].reset_index(drop=True)
Y_test = X_data.iloc[test_indices,-1].reset_index(drop=True)
'''


# here train the test data are same as we use entire data to build decesion tree 
#and test how many of the really satisfy
X_train = X_data.iloc[:,:-1].reset_index(drop=True)
X_test = X_data.iloc[:,:-1].reset_index(drop=True)
Y_train = X_data.iloc[:,-1].reset_index(drop=True)
Y_test = X_data.iloc[:,-1].reset_index(drop=True)

print()
print("Decision Tree by Information Gain : ")

# Get the decesion tree
tree = get_tree(X_train, Y_train, 0, None)

# Print the tree, starting level = 0
print_tree(tree,0)

# Get the accuracy details
correct, incorrect = test_accuracy(X_test, Y_test, tree)

#Print the Test Case Details					
print()
print()
print("Total Data Set : ", correct+incorrect)
print("Correct Prediction : ", correct)
print("Incorrect Prediction : ", incorrect)
print("Accuracy : ", 100*correct/(correct+incorrect), "%")
print()
