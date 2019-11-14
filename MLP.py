import numpy as np
from numpy import genfromtxt, random, exp, array, dot

def sigmoid_derivative(z):
    result = sigmoid(z) * (1 -sigmoid(z))
    return result

def sigmoid(z):
    result = 1.0 / (1.0 + np.exp(-z))
    return result

def feedforward(x,w1,w2,w3,w4,w5,b1,b2,b3,b4,b5):
    h1 = sigmoid(dot(x,w1)+b1)
    h2 = sigmoid(dot(h1,w2)+b2)
    h3 = sigmoid(dot(h2,w3)+b3)
    h4 = sigmoid(dot(h3,w4)+b4)
    out = sigmoid(dot(h4,w5)+b5)
    return (h1,h2,h3,h4,out)

def backpropagation():
    return 'b'

def treinamento(x, y, iteracoes, nNeuronios):

    (w1,b1) = inicializaWeB(dim,nNeuronios)
    (w2,b2) = inicializaWeB(nNeuronios,nNeuronios)
    (w3,b3) = inicializaWeB(nNeuronios,nNeuronios)
    (w4,b4) = inicializaWeB(nNeuronios,nNeuronios)
    (w5,b5) = inicializaWeB(nNeuronios,10)

    for iteracao in range(iteracoes):
        # Pass the training set through our neural network
        (h1,h2,h3,h4,out) = feedforward(x,w1,w2,w3,w4,w5,b1,b2,b3,b4,b5)

        # Calculate the error for layer 5 (The difference between the desired output
        # and the predicted output).
        layer5_error = y - out
        layer5_delta = layer5_error * sigmoid_derivative(out)

        # Calculate the error for layer 4 (The difference between the desired output
        # and the predicted output).
        layer4_error = layer5_delta.dot(w5.T)
        layer4_delta = layer4_error * sigmoid_derivative(h4)

        # Calculate the error for layer 3 (The difference between the desired output
        # and the predicted output).
        layer3_error = layer4_delta.dot(w4.T)
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
        layer4_adjustment = h3.T.dot(layer4_delta)
        layer5_adjustment = h4.T.dot(layer5_delta)

        # Adjust the weights.
        w1 += layer1_adjustment
        w2 += layer2_adjustment
        w3 += layer3_adjustment
        w4 += layer4_adjustment
        w5 += layer5_adjustment
    return (w1,w2,w3,w4,w5,b1,b2,b3,b4,b5)

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
(w1,w2,w3,w4,w5,b1,b2,b3,b4,b5) = treinamento(X,Y,1000,100) 

(xTest,yTest,dim,classe) = leArquivo('dataSetMultiCamadas/test.csv')
X = np.array(xTest)
Y = np.array(yTest)
(h1,h2,h3,h4,out) = feedforward(X,w1,w2,w3,w4,w5,b1,b2,b3,b4,b5)

v = 0
f = 0
resultPredict = 0
resultExpected = 0

""" for i in range(len(out)):
    maior = 0
    resultPredict = 0
    for j in range(len(out[i])):
        if out[i][j] > maior:
            maior = out[i][j]
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

for i in range(len(out)):
    resultPredict = np.where(out[i] == np.amax(out[i]))
    resultExpected = np.where(Y[i] == np.amax(Y[i]))

    if resultPredict == resultExpected:
        v += 1
    else:
        f += 1

print('Precisao ',v/(v+f) * 100, '%')






