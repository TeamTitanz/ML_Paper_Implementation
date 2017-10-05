import numpy as np
import pandas as pd
import pickle as p
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import cPickle as cpickle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os.path

a_value = 'a2'
p_value = 'p750'
inputArray = []
outputArray = []
prediction_file = open('predictions_' + a_value + '_' + p_value + '.pickle', 'r')
predictions = p.load(prediction_file)

# Load Node2Vec Trained Model
node2vec_model = KeyedVectors.load_word2vec_format('./emd/supShort_supShort.emd')

# enter file input name
mentionMap = 'MentionMap.txt'
currentPath = os.getcwd()

# Start reading the file in separate lines
with open(currentPath + '/' + mentionMap) as f:
    lines = f.readlines()
    # stripping the newline character
    lines = [x.strip() for x in lines]

max = 0
for line in lines:
    if (';') in line:
        document = line.split(';')[0]
        referencesList = line.split(';')[1].split(',')
        if max < len(referencesList):
            max = len(referencesList)
outputCount = 250

recall = 0.0
index = 0
# xData = []
# yData = []
for line in lines:

    inputElement = []
    outputElement = []
    # vI = []
    documentId = line.split(';')[0]
    documentId = documentId.lstrip("0")
    print documentId + "/" + "2500"
    referencesList = line.split(';')[1].split(',')

    if ('' in referencesList):
        referencesList = filter(lambda a: a != '', referencesList)

    if (documentId in referencesList):
        referencesList = filter(lambda a: a != documentId, referencesList)

    prediction = predictions[index]
    word2vec_model_output = node2vec_model.similar_by_vector(prediction, topn=outputCount)

    for a in prediction:
        # vI.append(a)
        inputElement.append(a)

    # create a list of the documents only, returned by the model. Remove the vector values
    modelReturnedDocumentList = []
    for i in range(0, len(word2vec_model_output)):
        modelReturnedDocumentList.append(str(word2vec_model_output[i][0]))

    # print vI, ",",
    # rData = []
    # b_I = []
    # print referencesList
    # print modelReturnedDocumentList
    for similarCase in modelReturnedDocumentList:
        # v_rI = []
        if (similarCase in referencesList):
            # b_I.append(1)
            outputElement.append(1)
        else:
            # b_I.append(0)
            outputElement.append(0)

        for b in node2vec_model.__getitem__(similarCase):
            # v_rI.append(b)
            inputElement.append(b)

            # rData.append(v_rI)
            # print v_rI

    # xData.append([vI, rData])
    # yData.append(b_I)
    # print xData[0]
    # print yData[0]
    inputArray.append(inputElement)
    outputArray.append(outputElement)
    # print inputArray[0]
    # print outputArray[0]
    index += 1

print len(inputArray)
print len(outputArray)

print "Saving input..."
input_file = open('input[X]_' + a_value + '_' + p_value + '.pickle', 'w')
p.dump(inputArray, input_file)
print "Input saved"

# save output
print "Saving output..."
output_file = open('output[Y]_' + a_value + '_' + p_value + '.pickle', 'w')
p.dump(outputArray, output_file)
print "Output saved"
