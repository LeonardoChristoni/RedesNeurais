import numpy as np
from numpy import genfromtxt, random, exp, array, dot

def sigmoid_derivative(z):
    result = sigmoid(z) * (1 -sigmoid(z))
    return result

def sigmoid(z):
    result = 1.0 / (1.0 + np.exp(-z))
    return result

def feedforward():
    return 'a'

def backpropagation():
    return 'b'

def treinamento(x, y, iteracoes):

    (w1,b1) = inicializaWeB(dim,1)
    (w2,b2) = inicializaWeB(1,1)
    (w3,b3) = inicializaWeB(1,1)

    for iteracao in range(iteracoes):
        # Pass the training set through our neural network
        h1 = sigmoid(dot(x,w1)+b1)
        h2 = sigmoid(dot(h1,w2)+b2)
        h3 = sigmoid(dot(h2,w3)+b3)

        # Calculate the error for layer 2 (The difference between the desired output
        # and the predicted output).
        layer2_error = y - h2
        layer2_delta = layer2_error * sigmoid_derivative(h2)

        # Calculate the error for layer 1 (By looking at the weights in layer 1,
        # we can determine by how much layer 1 contributed to the error in layer 2).
        layer1_error = layer2_delta.dot(w2.transpose())
        layer1_delta = layer1_error * sigmoid_derivative(h1)

        # Calculate how much to adjust the weights by
        layer1_adjustment = x.T.dot(layer1_delta)
        layer2_adjustment = h1.T.dot(layer2_delta)

        # Adjust the weights.
        w1 += layer1_adjustment
        w2 += layer2_adjustment
        print('Erro ultima camada')
        print(layer2_error)

  
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
    data = np.genfromtxt('datasets\multi-layer\LS_train.csv',delimiter=',',skip_header=2)
    dimClasse = np.genfromtxt('datasets\multi-layer\LS_train.csv',delimiter=',',skip_footer=len(data),dtype=str)
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

(x,y,dim,classe) = leArquivo('datasets\multi-layer\LS_train.csv')
X = np.array(x)
Y = np.array(y)
treinamento(X,Y,20)




