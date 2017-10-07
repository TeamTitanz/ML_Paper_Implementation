import cPickle as p

import numpy as np
from sklearn.neural_network import MLPRegressor

#####################Reading Files#######################################
dimensionsInput = 0
inputArr = []
docIdArr = []

dimensionsOutput = 0
outputArr = []
docIdArrNode2Vec = dict()
a_value = 'a2'
p_value = 'p750'

with open('./' + a_value + '/' + p_value + '/DocVector_in_word2vec.txt') as f:
    content = f.readlines()
    numberOfVectors = 0

    for i in range(len(content)):
        if i == 0:
            numberOfVectors = content[i].split(" ")[0]
            dimensionsInput = content[i].split(" ")[1]
        else:
            input = []
            data = content[i].split(" ")
            docIdArr.append(data[0])

            for j in range(int(dimensionsInput)):
                input.append(float(data[j + 1]))
            inputArr.append(input)

with open('./emd/supShort_supShort.emd') as f:
    content = f.readlines()
    numberOfVectors = 0
    dimensions = 0
    outputArr = []

    for i in range(len(content)):
        if i == 0:
            numberOfVectors = content[i].split(" ")[0]
            dimensionsOutput = content[i].split(" ")[1]
        else:
            output = []
            data = content[i].split(" ")

            for j in range(int(dimensionsOutput)):
                output.append(float(data[j + 1]))
            docIdArrNode2Vec[data[0]] = output

for i in range(len(docIdArr)):
    outputArr.append(docIdArrNode2Vec[docIdArr[i]])

################End Reading Files##############################

lvInput = np.array(inputArr)
lvTarget = np.array(outputArr)
print(lvTarget[0])

# for p250
##clf = MLPRegressor(solver='lbfgs', activation='logistic',alpha=0.001, hidden_layer_sizes=(250,), random_state=1, max_iter = 2000, learning_rate='adaptive'
##                   ,verbose = 'True')
##clf.fit(lvInput,lvTarget)

# for p500
##clf = MLPRegressor(solver='lbfgs', activation='logistic',alpha=0.001, hidden_layer_sizes=(250,), random_state=1, max_iter = 2000, learning_rate='adaptive'
##                   ,verbose = 'True')
##clf.fit(lvInput,lvTarget)

# for p750
clf = MLPRegressor(solver='lbfgs', activation='logistic', alpha=0.001, hidden_layer_sizes=(250,), random_state=1,
                   max_iter=2000, learning_rate='adaptive'
                   , verbose='True')
clf.fit(lvInput, lvTarget)

# for p2000
##clf = MLPRegressor(solver='lbfgs', activation='relu',alpha=0.001, hidden_layer_sizes=(200), random_state=1, max_iter = 2000, learning_rate='adaptive',
##                   learning_rate_init= 0.001, verbose = 'True')
##clf.fit(lvInput,lvTarget)

# for p1000
##clf = MLPRegressor(solver='lbfgs', activation='relu',alpha=0.001, hidden_layer_sizes=(130), random_state=1, max_iter = 2000, learning_rate='adaptive',
##                   learning_rate_init= 0.001, verbose = 'True')
##clf.fit(lvInput,lvTarget)

# save model file
model_file = open('model_' + a_value + '_' + p_value + '.pickle', 'wb')
p.dump(clf, model_file)

##    #open model file
##model_file = open('model.pickle', 'r') 
##clf = p.load(model_file)


# run predictions
predictions = clf.predict(lvInput)
print(predictions[0])

# save predictions
predictions_file = open('predictions_' + a_value + '_' + p_value + '.pickle', 'wb')
p.dump(predictions, predictions_file)
