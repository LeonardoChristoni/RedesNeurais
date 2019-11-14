import numpy as np
from numpy import genfromtxt, random, exp, array, dot

def sigmoid_derivative(z):
    result = sigmoid(z) * (1 -sigmoid(z))
    return result

def sigmoid(z):
    result = 1.0 / (1.0 + np.exp(-z))
    return result

def feedforward(x,w1,w2,w3,b1,b2,b3):
    h1 = sigmoid(dot(x,w1)+b1)
    h2 = sigmoid(dot(h1,w2)+b2)
    h3 = sigmoid(dot(h2,w3)+b3)
    return (h1,h2,h3)

def backpropagation():
    return 'b'

def treinamento(x, y, iteracoes):

    (w1,b1) = inicializaWeB(dim,1)
    (w2,b2) = inicializaWeB(1,1)
    (w3,b3) = inicializaWeB(1,10)

    for iteracao in range(iteracoes):
        # Pass the training set through our neural network
        (h1,h2,h3) = feedforward(x,w1,w2,w3,b1,b2,b3)

        # Calculate the error for layer 3 (The difference between the desired output
        # and the predicted output).
        layer3_error = y - h3
        layer3_delta = layer3_error * sigmoid_derivative(h3)

        # Calculate the error for layer 2 (The difference between the desired output
        # and the predicted output).
        layer2_error = layer3_delta.dot(w3.T)
        layer2_delta = layer2_error * sigmoid_derivative(h2)

        # Calculate the error for layer 1 (By looking at the weights in layer 1,
        # we can determine by how much layer 1 contributed to the error in layer 2).
        layer1_error = layer2_delta.dot(w2.transpose())
        layer1_delta = layer1_error * sigmoid_derivative(h1)

        # Calculate how much to adjust the weights by
        layer1_adjustment = x.T.dot(layer1_delta)
        layer2_adjustment = h1.T.dot(layer2_delta)
        layer3_adjustment = h2.T.dot(layer3_delta)

        # Adjust the weights.
        w1 += layer1_adjustment
        w2 += layer2_adjustment
        w3 += layer3_adjustment
        #print('Erro ultima camada')
        #print(layer2_error)
    return (w1,w2,w3,b1,b2,b3)

def inicializaWeB(dim,numNeuronios):
    w = []
    b = []
    for i in range(int(dim)):
        wLinha = []
        b = []
        for j in range(int(numNeuronios)):
            b.append(random.uniform(-1,1))
            wLinha.append(random.uniform(-1,1))
        w.append(wLinha)

    W = np.array(w)
    B = np.array(b)
    return (W,B)

def leArquivo(path):
    data = np.genfromtxt(path,delimiter=',',skip_header=2)
    dimClasse = np.genfromtxt(path,delimiter=',',skip_footer=len(data),dtype=str)
    dim = dimClasse[0].split(':')[1]
    classe = dimClasse[1].split(':')[1]

    x = []
    y = []
    for i in range(len(data)):
        linha = data[i]
        xLinha = []
        yLinha = []
        for k in range(len(linha)):
            if k < int(dim):
                xLinha.append(linha[k])
            else:
                yLinha.append(linha[k])
        x.append(xLinha)
        y.append(yLinha)

    return (x,y,dim,classe)

(xTrain,yTrain,dim,classe) = leArquivo('dataSetMultiCamadas/train.csv')
X = np.array(xTrain)
Y = np.array(yTrain)
(w1,w2,w3,b1,b2,b3) = treinamento(X,Y,20) 

(xTest,yTest,dim,classe) = leArquivo('dataSetMultiCamadas/test.csv')
X = np.array(xTest)
Y = np.array(yTest)
(h1,h2,h3) = feedforward(X,w1,w2,w3,b1,b2,b3)

v = 0
f = 0
resultPredict = 0
resultExpected = 0

""" for i in range(len(h3)):
    maior = 0
    resultPredict = 0
    for j in range(len(h3[i])):
        if h3[i][j] > maior:
            maior = h3[i][j]
            resultPredict = j

    maior = 0
    resultExpected = 0 
    for k in range(len(Y[i])):
        if Y[i][k] > maior:
            maior = Y[i][k]
            resultExpected = k 
    
    if resultPredict == resultExpected:
        v += 1
    else:
        f += 1 """

for i in range(len(h3)):
    resultPredict = np.where(h3[i] == np.amax(h3[i]))
    resultExpected = np.where(Y[i] == np.amax(Y[i]))

    if resultPredict == resultExpected:
        v += 1
    else:
        f += 1

print('Precisao ',v/(v+f) * 100, '%')






