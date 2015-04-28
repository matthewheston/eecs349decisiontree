import sys
import csv
import random
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def csvToDict(filename):
	with open(filename, 'rU') as f:
		# Change DictReader to list (assumes it can fit in memory)
		dataDict = list(csv.DictReader(f, delimiter=','))
	return dataDict

def createFeatures(dataFiles, continuousAttrs, classLabel):
	''' Takes a list of CSV dictionary objects and a list of continuous attributes, and returns a
	list composed of a dictionary of binary features for each item.
	Continuous attributes are converted to quartile representations, and then all attributes
	are converted into binary features.'''

	# Transform continuous variables into strings representing quartiles, and create a
	# dictionary that maps attribute names to the set of discrete values used.

	# Initialize both dictionaries using the first row
	data = dataFiles[0]
	firstRow = data[0]
	continuousDict = {k:[] for k in firstRow if k in continuousAttrs}
	discreteDict = {k:set([]) for k in firstRow if k not in continuousAttrs}

	# Go through the data, and create the dictionaries
	for row in data:
		for attribute in row:
			if attribute in continuousAttrs:
				continuousDict[attribute].append(strToFloat(row[attribute]))
			else:
				discreteDict[attribute].add(row[attribute])

	# Change the continuous attributes to discrete quartiles, in the original data
	quartilesDict = getQuartiles(continuousDict)
	dataFiles, newStrings = discretizeContinuous(dataFiles, quartilesDict)
	# Update discrete dictionary to include quartile info.
	discreteDict.update({k:set(v) for k,v in newStrings.items()})
	# Create a new version of the data, with binary discrete features for each possible
	# value for each original attribute
	binaryDataFiles = []
	for data in dataFiles:
		binaryData = []
		for row in data:
			currRow = {}
			for attribute, value in row.items():
				if attribute != classLabel:
					for feature in discreteDict[attribute]:
						featureName = '{}_is_{}'.format(attribute, feature)
						if value == feature:
							currRow[featureName] = True
						else:
							currRow[featureName] = False
				# We let the class label keep its original value.
				else:
					currRow[attribute] = value
			binaryData.append(currRow)
		binaryDataFiles.append(binaryData)
	return binaryDataFiles
	

def getQuartiles(cDict):
	'''Takes a dictionary of lists of continuous values, and returns a dictionary of lists of
	quartile values for each item in the dictionary'''

	finalDict = {}
	numQuantiles = 4
	for attr in cDict:
		quantList = []
		n = len(cDict[attr])+1
		sortedVals = sorted(cDict[attr])
		for i in range(numQuantiles-1):
			# Append the value in the list that is located at the ith quantile,
			# if it divides, evenly, otherwise, average the quartile before and after
			firstLoc = n*(i+1)/numQuantiles - 1
			if n*(i+1) % numQuantiles == 0:
				quantVal = sortedVals[firstLoc]
			else:
				quantVal = (sortedVals[firstLoc]+sortedVals[firstLoc + 1])/2.0
			if quantVal not in quantList:
				quantList.append(quantVal)
		finalDict[attr] = quantList
	return finalDict

def discretizeContinuous(dataFiles, qDict):
	'''Takes a list of data files, and a dictionary of sorted quartile lists, and 
	updates the data dictionaries to discrete values, based on quartiles, and returns 
	the strings used to create the new discrete attributes'''
	for data in dataFiles:
		for row in data:
			# For each row in the data, go through each of the possible values, and compare
			# them to the actual value, then replace with a new discrete string.
			for attr, vals in qDict.items():
				qIndex = 0
				qEnd = len(vals)-1
				for quantile in vals:
					if float(row[attr]) < quantile:
						if qIndex == 0:
							# If this is the first value, then it's less than
							row[attr] = '<{}'.format(quantile)
						else:
							# otherwise, it's greater than the earlier value
							row[attr] = '>={} and <{}'.format(vals[qIndex-1], quantile)
						break
					else:
						# If it's greater than the last item
						if qIndex == qEnd:
							row[attr] = '>={}'.format(quantile)
					qIndex += 1
		
	# Get names of new strings
	stringDict = {}
	for attr, vals in qDict.items():
		qVals = sorted(vals)
		newStrings = ['<{}'.format(qVals[0]),'>={}'.format(qVals[-1])]
		newStrings += ['>={} and <{}'.format(qVals[i-1], qVals[i]) for i in range(1,len(qVals))]
		stringDict[attr] = newStrings
	return dataFiles, stringDict

def strToFloat(s):
	try:
		return float(s)
	except:
		return s

def partitionData(binaryData, n):
	'''Takes a list, randomizes it, then returns a list of n lists of subsets of the data'''
	# Randomize binary data, and put into partitions
	random.shuffle(binaryData)
	partitionList = []
	dataSize = len(binaryData)
	startOfRange = 0
	for x in range(n):
		endOfRange = dataSize*(x+1)/n
		partitionList.append(binaryData[startOfRange:endOfRange])
		startOfRange = endOfRange
	return partitionList


def bestFeature(bData, classLabel):
	'''Takes a partitioned binary data file and a class label. Returns the feature 
	with the highest chiSquare score'''
	# Create an observation dictionary for each partition
	obsDict = {}
	# For each row, update the count for the cross of classLabel value and attribute value
	for r in bData:
		# Find out which value the class label has, and store it
		classVal = r[classLabel]
		for attr, val in r.items():
			if attr != classLabel:
				# Initialize the dictionaries, if they don't exist
				obsDict[attr] = obsDict.get(attr, {})
				obsDict[attr][classVal] = obsDict[attr].get(classVal, [0,0])
				if val == True:
					obsDict[attr][classVal][0] += 1
				else:
					obsDict[attr][classVal][1] += 1
	# Use the observation counts to calculate chi square scores
	chiDict = {}
	# Format dictionary in the way the calcChiScore function is expecting it.
	for attr, val in obsDict.items():
		chiDict[attr] = calcChiScore([[n for n in x] for x in val.values()])
	# Get the max feature, pull out the name of the original attribute and its value
	# then return the best feature, and the score, for each partition.
	bestFeature = max(chiDict, key=chiDict.get)
	# return the name of the best feature, and the Chi Square score
	return bestFeature, chiDict[bestFeature]

def calcChiScore(m):
	'''Takes a list of 2 rows of binary observation matrix, and returns the Chi-squared score'''
	chiScore = 0
	try:
		# Get the total number of observations
		totalObs = float(sum(m[0]) + sum(m[1]))
		for i in range(2):
			for j in range(2):
				# For each item in the matrix, calculate the expected value,
				# followed by the chiScore, and sum chi scores
				expectedVal = (m[j][i]+m[(j+1)%2][i])*(m[j][i] + m[j][(i+1)%2])/totalObs
				chiScore += (m[j][i] - expectedVal)**2 / expectedVal
	except:
		return 0
	return chiScore

def makeDecisionTree(data, attributes, classLabel, threshold=3.841):
	'''Takes a list of dictionaries, a list of attributes, and a classlabel, returns
	a decision tree'''
	data = data[:]
	default = getMajorityVal(data, classLabel)

	# If there's no data, or no attributes left to pick from, then return the majority value
	if len(data) == 0 or len(attributes)-1 == 0:
		return default
	# If all the data has the same class label, then return that class label
	if sameClassLabel(data, classLabel):
		return default
	else:
		# Get the best new split
		currAttr, chiSq = bestFeature(data, classLabel)
		if chiSq > threshold:
			#Initialize the tree
			tree = {currAttr:{}}
			# Split and recurse
			leftData = [r for r in data if r[currAttr] == True]
			rightData = [r for r in data if r[currAttr] == False]
			# Eliminate the current attribute from the set
			newAttrs = [a for a in attributes if a != currAttr]
			lTree = makeDecisionTree(leftData, newAttrs, classLabel, threshold)
			rTree = makeDecisionTree(rightData, newAttrs, classLabel, threshold)
			# Add the subtrees to the tree dictionary, under the correct binary value
			tree[currAttr][True] = lTree
			tree[currAttr][False] = rTree
		else:
			return default
	return tree



def getMajorityVal(data, classLabel, debug=False):
	'''Takes a list of dictionaries and a class label, returns the majority value of the classlabel'''
	default = 'Y'
	labelCount = {}
	if len(data) == 0:
		return default
	for row in data:
		val = row[classLabel]
		labelCount[val] = labelCount.get(val, 0) + 1
	return max(labelCount, key=labelCount.get) 

def sameClassLabel(data, label):
	'''Takes a data file and a class label, and returns True if all labels are the same.
	Otherwise, returns False'''
	firstLabel = data[0][label]
	for row in data:
		if row[label] != firstLabel:
			return False
	return True

def scoreTree(tree, data, classLabel):
	'''Takes a decision tree, a test file, and a classLabel; returns the Zero-One Loss'''
	predictionList = getPredictions(tree, data, classLabel)
	wrongCount = 0
	wrongList = []
	# Count the wrong scores
	for p in predictionList:
		if p[0] != p[1]:
			wrongCount += 1
			wrongList.append((p[0],p[1]))
	# Normalize by number of scores
	return wrongCount / float(len(predictionList))

def getPredictions(tree, data, classLabel):
	'''Takes a decision tree, a dataset, and a class label. Returns a list of tuples, in 
	the form (predictedClass, actualClass)'''
	data = data[:]
	if len(data) == 0:
		return []
	predictions = []
	for feature in tree:
		# If the feature is a dictionary
		try:
			# Split into 2 datasets (assumes binary features)
			lData = [x for x in data if x[feature] == True]
			rData = [x for x in data if x[feature] == False]
			# Recurse, and add predictions from both halves
			predictions = predictions + getPredictions(tree[feature][False], rData, classLabel)
			predictions = predictions + getPredictions(tree[feature][True], lData, classLabel)
		except KeyError:
			# If the feature isn't a dictionary, then it's a label.
			# Add a tuple for the prediction and the actual label for everything in data.
			predictions = [(feature, x[classLabel]) for x in data]
	return predictions

def getCrossValResults(data, kFold, classLabel, sampleSizes=None, thresholds = [3.841]):
	'''Takes a data file, the number of kFold partitions, and a class label. Returns a dictionary
	of zero-one loss score averages for each condition'''
	# Initialize sample sizes, if a value hasn't been passed
	if not sampleSizes: sampleSizes = [len(data)]
	attributes = data[0].keys()
	partitions = partitionData(data, kFold)
	xValData = makeCrossValData(partitions)
	results = {}
	# For each sample size, and each threshold value, get results
	for s in sampleSizes:
		for t in thresholds:
			scores = []
			for d in xValData:
				# Get the right sample size from the beginning of the training data
				trainingData = d[0][:s]
				# Get the test data
				testData = d[1]
				# Make a tree
				dTree = makeDecisionTree(trainingData, attributes, classLabel, t)
				# Score the tree, using the test data
				score = scoreTree(dTree, testData, classLabel)
				scores.append(score)
			# Average the scores for each kfold set.
			results[(s,t)] = sum(scores)/float(len(scores))
	return results

def makeCrossValData(partitions):
	'''Takes a list of partitioned data, and returns a list of tuples in the form
	(trainingData, testData), where the trainingData has been randomized'''
	results = []
	for i in range(len(partitions)):
		trainData = []
		# Put all of the partitions that aren't in the test data in the training data.
		for j in range(len(partitions)):
			if i != j:
				trainData += partitions[j]
		# Randomize the training data again
		random.shuffle(trainData)
		results.append((trainData, partitions[i]))
	return results


def main():
    # Get Parameters
    trainingData = pd.read_csv(sys.argv[1])
    validationData = pd.read_csv(sys.argv[2])
    toPrune = sys.argv[3]
    percentDataToUse = float(sys.argv[4])/100

    # Make Tree
    rows = np.random.choice(trainingData.index.values, percentDataToUse * len(trainingData),
            replace = False)
    tree = make_decision_tree(trainingData


    # Prune Tree

    # Plot performance

	if len(sys.argv) == 3:
		continuousAttrs = ["Latitude", "Longitude", "ReviewCount"]
		classLabel = "GoodForLunch"
		filename = sys.argv[1]
		testData = sys.argv[2]
		kFoldVal = 10
				
		# Convert data to dictionaries
		origData = csvToDict(filename)
		if testData == '--xVal':
			sampleList = [20,40,60,120,180]
			# Get binary data
			bData = createFeatures([origData], continuousAttrs, classLabel)[0]
			results = getCrossValResults(bData, kFoldVal, classLabel, sampleSizes=sampleList)
			print results
				
		elif testData == '--thresh':
			threshList = [1.323, 2.706, 5.024, 10.828]
			sampleList = [20,40,60,120,180]
			# Get binary data
			bData = createFeatures([origData], continuousAttrs, classLabel)[0]
			results = getCrossValResults(bData, kFoldVal, classLabel, sampleList, threshList)
			print results

		#In the normal situation, w/2 data files
		else:
			tData = csvToDict(testData)
			# Create the binary version of the data files
			bDataFiles = createFeatures([origData, tData], continuousAttrs, classLabel)
			binaryData = bDataFiles[0]
			binaryTData = bDataFiles[1]
			# Create decision tree
			attributeList = binaryData[0].keys()
			decisionTree = makeDecisionTree(binaryData, attributeList, classLabel)
			# Analyze test data
				
			score = scoreTree(decisionTree, binaryTData, classLabel)
			print "ZERO-ONE LOSS={}".format(score)			
	else:
		print 'inputs are TrainingData and TestData'	



if __name__ == "__main__":
	main()
