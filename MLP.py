import numpy as np
from numpy import genfromtxt, random, exp, array, dot

def sigmoid_derivative(z):
    result = z * (1 - z)
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

def treinamento(x, y, iteracoes, nNeuronios):

    (w1,b1) = inicializaWeB(dim,nNeuronios)
    (w2,b2) = inicializaWeB(nNeuronios,nNeuronios)
    (w3,b3) = inicializaWeB(nNeuronios,nNeuronios)
    (w4,b4) = inicializaWeB(nNeuronios,nNeuronios)
    (w5,b5) = inicializaWeB(nNeuronios,10)
    
    for iteracao in range(iteracoes):
        print(f"\r{iteracao+1}/{iteracoes}", end='', flush=True)

        #feedforward - retorno saida das 5 camadas
        (h1,h2,h3,h4,out) = feedforward(x,w1,w2,w3,w4,w5,b1,b2,b3,b4,b5)

        erroQuadratico = ((y-out)**2)/2
        #print("Erro quadratico ",erroQuadratico)

        #calcula erro e gradiente da camada 5
        layer5_error = y - out
        layer5_delta = layer5_error * sigmoid_derivative(out)

        #calcula erro e gradiente da camada 4
        layer4_error = layer5_delta.dot(w5.T)
        layer4_delta = layer4_error * sigmoid_derivative(h4)

        #calcula erro e gradiente da camada 3
        layer3_error = layer4_delta.dot(w4.T)
        layer3_delta = layer3_error * sigmoid_derivative(h3)

        #calcula erro e gradiente da camada 2
        layer2_error = layer3_delta.dot(w3.T)
        layer2_delta = layer2_error * sigmoid_derivative(h2)

        #calcula erro e gradiente da camada 1
        layer1_error = layer2_delta.dot(w2.transpose())
        layer1_delta = layer1_error * sigmoid_derivative(h1)

        lr = 0.01

        # Calculate how much to adjust the weights by
        layer1_adjustment = x.T.dot(layer1_delta)
        layer2_adjustment = h1.T.dot(layer2_delta)
        layer3_adjustment = h2.T.dot(layer3_delta)
        layer4_adjustment = h3.T.dot(layer4_delta)
        layer5_adjustment = h4.T.dot(layer5_delta)

        # atualiza pesos
        w1 -= lr * layer1_adjustment
        w2 -= lr * layer2_adjustment
        w3 -= lr * layer3_adjustment
        w4 -= lr * layer4_adjustment
        w5 -= lr * layer5_adjustment

        # atualiza bias
    #    b1 -= lr * layer1_delta
    #    b2 -= lr * layer2_delta
    #    b3 -= lr * layer3_delta
    #    b4 -= lr * layer4_delta
    #    b5 -= lr * layer5_delta
    return (w1,w2,w3,w4,w5,b1,b2,b3,b4,b5)

def inicializaWeB(dim,numNeuronios):
    w = []
    b = []

    #dim é a numero de entrada
    for i in range(int(dim)):
        wLinha = []
        b = []
        #varios neuronios para cada entrada
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

def gravaArquivoNN(nomeArquivo,layers, biases):
    with open(nomeArquivo, mode='w') as arquivo:
        i = 1
        for l, b in zip(layers, biases):
            arquivo.write(f"camada{i}\n")
            arquivo.write(f"entrada {l.shape[1]}\n")
            arquivo.write(f"saida  {l.shape[0]}\n")
            arquivo.write('W\n')
            for line in l:
                for entradaParcial in line:
                    arquivo.write(f' {entradaParcial}')
                arquivo.write('\n')    

            arquivo.write('b\n')
            for element in b:
                arquivo.write(f' {element}')
            arquivo.write('\n')
            arquivo.write('ativacao sigmoid\n')
            arquivo.write('--\n')

            i += 1


#--------------------------------MAIN---------------------------------#
(xTrain,yTrain,dim,classe) = leArquivo('mnist_train.csv')
X = np.array(xTrain)
Y = np.array(yTrain)

(w1,w2,w3,w4,w5,b1,b2,b3,b4,b5) = treinamento(X,Y,350000,40) 
(h1,h2,h3,h4,out) = feedforward(X,w1,w2,w3,w4,w5,b1,b2,b3,b4,b5)

(xTest,yTest,dim,classe) = leArquivo('mnist_test.csv')
X = np.array(xTest)
Y = np.array(yTest)

wFinal = []
wFinal.append(w1)
wFinal.append(w2)
wFinal.append(w3)
wFinal.append(w4)
wFinal.append(w5)

bFinal = []
bFinal.append(b1)
bFinal.append(b2)
bFinal.append(b3)
bFinal.append(b4)
bFinal.append(b5)

(h1,h2,h3,h4,out) = feedforward(X,w1,w2,w3,w4,w5,b1,b2,b3,b4,b5)

v = 0
f = 0
resultPredict = 0
resultExpected = 0

tp = 0
tn = 0
fp = 0
fn = 0

for i in range(len(out)):
    maior = 0
    resultPredict = 0
    #busca o valor previsto
    for j in range(len(out[i])):
        if out[i][j] > maior:
            maior = out[i][j]
            resultPredict = j

    maior = 0
    resultExpected = 0 
    #busca o valor correto, esperado
    for k in range(len(Y[i])):
        if Y[i][k] > maior:
            maior = Y[i][k]
            resultExpected = k 
    
    for x in range(10):
        if resultPredict == x :
            if resultExpected == x:
                tp += 1
            else :
                fp += 1
        else:
            if resultExpected == x:
                fn += 1
            else :
                tn += 1

print('\nPrecisao ',tp/(tp+fp) * 100, '%')
print('Recall ',tp/(tp+fn) * 100, '%')
print('Acuracia ',(tp+tn)/(tp+fp+tn+fn) * 100, '%')
print('tp ',tp)
print('tn ',tn)
print('fp ',fp)
print('fn ',fn)

gravaArquivoNN("saida_NN.txt",wFinal,bFinal)





