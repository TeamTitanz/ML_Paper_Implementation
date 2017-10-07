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

prediction_file = open('predictions_a2_750.pickle', 'r')
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
xData = []
yData = []
for line in lines:
    # if semicolon is not there, there are no reference
    if (';') not in line:
        pass

    else:
        vI = []
        documentId = line.split(';')[0]
        documentId = documentId.lstrip("0")
        referencesList = line.split(';')[1].split(',')

        if ('' in referencesList):
            referencesList = filter(lambda a: a != '', referencesList)

        if (documentId in referencesList):
            referencesList = filter(lambda a: a != documentId, referencesList)

        word2vec_model_output = node2vec_model.similar_by_vector(predictions[index], topn=outputCount)

        for a in predictions[index]:
            vI.append(a)

            # create a list of the documents only, returned by the model. Remove the vector values
            modelReturnedDocumentList = []
        for i in range(0, len(word2vec_model_output)):
            modelReturnedDocumentList.append(str(word2vec_model_output[i][0]))

        # print vI, ",",
        rData = []
        b_I = []
        # print referencesList
        # print modelReturnedDocumentList
        for similarCase in modelReturnedDocumentList:
            v_rI = []
            if (similarCase in referencesList):
                b_I.append(1)
            else:
                b_I.append(0)

            for b in node2vec_model.__getitem__(similarCase):
                v_rI.append(b)
            rData.append(v_rI)
            # print v_rI

        xData.append([vI, rData])
        yData.append(b_I)

    print xData[0]
    # print yData[0]

    index += 1
    break

    # print (recall/2500.0)*100.0
