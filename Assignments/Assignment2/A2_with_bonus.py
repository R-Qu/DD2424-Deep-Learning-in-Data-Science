 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-
 # author: RuiQu  rqu@kth.se

import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import floor, sqrt
from tqdm import tqdm

N = 10000
d = 3072
K = 10
cifar10_labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def createOneHotLabels(labels):
    one_hot_labels = np.zeros((N, K))
    for i in range(len(labels)):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels

def loadBatch(filename):
    data = np.zeros((N, d))
    labels = np.zeros((N, 1))
    dict = unpickle(filename)
    data = dict[bytes("data", 'utf-8')] / 255.0  #Normalization
    labels = np.array(dict[bytes("labels", 'utf-8')])
    one_hot_labels = createOneHotLabels(labels)

    return data.T, one_hot_labels.T, labels

def loadDataset():
    trainSet = {}
    testSet = {}
    validationSet = {}

    for i in [1, 3, 4, 5]:
        t1, t2, t3 = loadBatch("dataset/data_batch_" + str(i))
        if i == 1:
            trainSet["data"] = t1
            trainSet["one_hot"] = t2
            trainSet["labels"] = t3
        else:
            trainSet["data"] = np.column_stack((trainSet["data"], t1))
            trainSet["one_hot"] = np.column_stack((trainSet["one_hot"], t2))
            trainSet["labels"] = np.append(trainSet["labels"], t3)

    a, b, c = loadBatch("dataset/data_batch_2")

    validationSet["data"], validationSet["one_hot"], validationSet["labels"] = a[:, :1000], b[:, :1000], c[:1000]

    trainSet["data"] = np.column_stack((trainSet["data"], a[:, 1000:]))
    trainSet["one_hot"] = np.column_stack((trainSet["one_hot"], b[:, 1000:]))
    trainSet["labels"] = np.append(trainSet["labels"], c[1000:])
    testSet["data"], testSet["one_hot"], testSet["labels"] = loadBatch("dataset/test_batch")

    temp = np.copy(trainSet["data"]).reshape((32, 32, 3, 49000), order='F')
    temp = np.flip(temp, 0)
    temp = temp.reshape((3072, 49000), order='F')

    trainSet["data"] = np.column_stack((trainSet["data"], temp))
    trainSet["one_hot"] = np.column_stack((trainSet["one_hot"], trainSet["one_hot"]))
    trainSet["labels"] = np.append(trainSet["labels"], trainSet["labels"])

    #Z-scores Normalization
    mean = np.mean(trainSet["data"], axis=1)
    mean = mean[:, np.newaxis]
    std = np.std(trainSet["data"], axis=1)
    std = std[:, np.newaxis]

    trainSet["data"] = (trainSet["data"] - mean)/std
    validationSet["data"] = (validationSet["data"] - mean)/std
    testSet["data"] = (testSet["data"] - mean)/std

    return trainSet, validationSet, testSet

class Classifier():
    def __init__(self, decay_eta, cyclical_eta, regularization_term, batch_size, n_epochs, weight_decay, shuffling, hidden_nodes, rho, leaky_RELU):
        self.W2 = np.zeros((K, hidden_nodes))
        self.W1 = np.zeros((hidden_nodes, d))
        self.b2 = np.zeros((K, 1))
        self.b1 = np.zeros((hidden_nodes, 1))

        self.W2_momentum = np.zeros((K, hidden_nodes))
        self.W1_momentum = np.zeros((hidden_nodes, d))
        self.b2_momentum = np.zeros((K, 1))
        self.b1_momentum = np.zeros((hidden_nodes, 1))

        self.decay_eta = decay_eta
        self.cyclical_eta = cyclical_eta
        self.eta = 1e-5
        self.eta_min = 1e-5
        self.eta_max = 2e-3

        self.lambda_reg = regularization_term
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.shuffling = shuffling
        self.hidden_nodes = hidden_nodes
        self.rho = rho

        self.leaky_RELU = leaky_RELU

        np.random.seed(1)

        self.initialization()

    def initialization(self):
        mu = 0
        sigma = sqrt(2) / sqrt(d)

        self.W1 = np.random.normal(mu, sigma, self.W1.shape)
        self.W2 = np.random.normal(mu, sigma, self.W2.shape)
        self.b2 = np.zeros((K, 1))
        self.b1 = np.zeros((self.hidden_nodes, 1))

        self.W2_momentum = np.zeros((K, self.hidden_nodes))
        self.W1_momentum = np.zeros((self.hidden_nodes, d))
        self.b2_momentum = np.zeros((K, 1))
        self.b1_momentum = np.zeros((self.hidden_nodes, 1))

    def evaluateClassifier(self, X, W1, b1, W2, b2):
        s1 = np.dot(W1, X) + b1
        if (self.leaky_RELU):
            h = np.maximum(s1, 0.01 * s1)
        else:
            h = np.maximum(s1, 0)
        s2 = np.dot(W2, h) + b2
        P = self.softmax(s2)
        assert(P.shape == (K, X.shape[1]))
        return P

    def softmax(self, x):
        r = np.exp(x) / sum(np.exp(x))
        return r

    def computeCost(self, X, Y, W1, b1, W2, b2):
        regularization = self.lambda_reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        loss_sum = 0
        for i in range(X.shape[1]):
            x = np.zeros((d, 1))
            y = np.zeros((K, 1))
            x = X[:, [i]]
            y = Y[:, [i]]
            loss_sum += self.cross_entropy(x, y, W1, b1, W2, b2)
        loss_sum /= X.shape[1]
        final = loss_sum + regularization
        assert(len(final) == 1)
        return final, loss_sum

    def cross_entropy(self, x, y, W1, b1, W2, b2):
        l = - np.log(np.dot(y.T, self.evaluateClassifier(x, W1, b1, W2, b2)))
        assert(len(l) == 1)
        return l

    def computeAccuracy(self, X, Y):
        acc = 0
        for i in range(X.shape[1]):
            P = self.evaluateClassifier(X[:, [i]], self.W1, self.b1, self.W2, self.b2)
            label = np.argmax(P)
            if label == Y[i]:
                acc += 1
        acc /= X.shape[1]
        return acc

    def compute_gradients(self, X, Y, P, W1, W2, b1):
        S1 = np.dot(W1, X) + b1

        if (self.leaky_RELU):
            H = np.maximum(S1, 0.01 * S1)
        else:
            H = np.maximum(S1, 0)

        G = -(Y.T - P.T).T

        gradb2 = np.mean(G, axis=-1, keepdims=True)
        gradW2 = np.dot(G, H.T)
        G = np.dot(G.T, W2)
        if (self.leaky_RELU):
            S1 = np.where(S1 > 0, 1, 0.01)
        else:
            S1 = np.where(S1 > 0, 1, 0)
        G = np.multiply(G.T, S1)
        gradb1 = np.mean(G, axis=-1, keepdims=True)
        gradW1 = np.dot(G, X.T)

        gradW1 /= X.shape[1]
        gradW2 /= X.shape[1]
        gradW1 += 2 * self.lambda_reg * W1
        gradW2 += 2 * self.lambda_reg * W2

        self.W1_momentum = self.W1_momentum * self.rho + self.eta * gradW1
        self.W2_momentum = self.W2_momentum * self.rho + self.eta * gradW2
        self.b1_momentum = self.b1_momentum * self.rho + self.eta * gradb1
        self.b2_momentum = self.b2_momentum * self.rho + self.eta * gradb2

    def datashuffling(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def fit(self, X, Y, validationSet=[]):
        n = X.shape[1]
        costsTraining = []
        costsValidation = []
        lossTraining = []
        lossValidation = []
        eta = []

        bestW1 = np.copy(self.W1)
        bestb1 = np.copy(self.b1)
        bestW2 = np.copy(self.W2)
        bestb2 = np.copy(self.b2)
        bestVal = self.computeCost(validationSet["data"], validationSet["one_hot"], self.W1, self.b1, self.W2, self.b2)[0]
        bestEpoch = 0
        n_batch = floor(n / self.batch_size)

        for i in tqdm(range(self.n_epochs)):

            #Decaying Learning Rate: self.eta = self.weight_decay * self.eta
            if (self.decay_eta):
                if i%10 == 0:
                    if i != 0:
                        self.eta = 0.1 * self.eta

            if (self.shuffling):
                X, Y = self.datashuffling(X.T, Y.T)
                X = X.T
                Y = Y.T

            for j in range(n_batch):

                j_start = j * self.batch_size
                j_end = (j + 1) * self.batch_size
                if j == n_batch - 1:
                    j_end = n

                Xbatch = X[:, j_start:j_end]
                Ybatch = Y[:, j_start:j_end]
                
                #Cylical Learning Rate
                if (self.cyclical_eta):
                    if i%2==0:
                        self.eta= (self.eta_max-self.eta_min)*j/n_batch+self.eta_min
                    else:
                        self.eta= (self.eta_min-self.eta_max)*j/n_batch+self.eta_max

                Pbatch = self.evaluateClassifier(Xbatch, self.W1, self.b1, self.W2, self.b2)
                self.compute_gradients(Xbatch, Ybatch, Pbatch, self.W1, self.W2, self.b1)

                self.W1 -= self.W1_momentum
                self.b1 -= self.b1_momentum
                self.W2 -= self.W2_momentum
                self.b2 -= self.b2_momentum

                eta.append(self.eta)
                #costsTraining.append((self.computeCost(Xbatch, Ybatch, self.W1, self.b1, self.W2, self.b2)[0])[0])
                
            val = (self.computeCost(validationSet["data"], validationSet["one_hot"], self.W1, self.b1, self.W2, self.b2)[0])[0]
            print("validation cost:"+str(val))
            vall = (self.computeCost(validationSet["data"], validationSet["one_hot"], self.W1, self.b1, self.W2, self.b2)[1])[0]
            print("validation loss:"+str(vall))
            if val < bestVal:
                bestVal = np.copy(val)
                bestW1 = np.copy(self.W1)
                bestb1 = np.copy(self.b1)
                bestW2 = np.copy(self.W2)
                bestb2 = np.copy(self.b2)
                bestEpoch = np.copy(i)

            costsTraining.append((self.computeCost(X, Y, self.W1, self.b1, self.W2, self.b2)[0])[0])
            costsValidation.append(val)

            lossTraining.append((self.computeCost(X, Y, self.W1, self.b1, self.W2, self.b2)[1])[0])
            lossValidation.append(vall)
            #accTraining.append(self.computeAccuracy(X,Y))
            #accValidation.append(self.computeAccuracy(validationSet["data"],validationSet["labels"]))

        print("Final cost: " + str(val))
       
        self.W1 = np.copy(bestW1)
        self.b1 = np.copy(bestb1)
        self.W2 = np.copy(bestW2)
        self.b2 = np.copy(bestb2)

        print("Best epoch: " + str(bestEpoch))
        print("Best cost: " + str(self.computeCost(validationSet["data"], validationSet["one_hot"], self.W1, self.b1, self.W2, self.b2)[0]))
        
        #Learning rate plot

        plt.plot(eta) 
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.title('Learning rate set over the epochs')
        plt.savefig("eta.png")
        plt.show()

        #Cost plot
        plt.plot(costsTraining,label="Training cost")
        plt.plot(costsValidation,label="Validation cost")
        plt.xlabel('Epoch number')
        plt.ylabel('Cost')
        plt.title('Cost for the training and validation set over the epochs')
        plt.legend(loc='best')
        plt.savefig("cost.png")
        plt.show()

        
        #Loss plot
        plt.plot(lossTraining, label="Training loss")
        plt.plot(lossValidation, label="Validation loss") 
        plt.xlabel('Epoch number')
        plt.ylabel('Loss')
        plt.title('Loss for the training and validation set over the epochs')
        plt.legend(loc='best')
        plt.savefig("loss.png")
        plt.show()
        
    def random_search(self, X, Y, validationSet, search_number, etas, lambdas):
        bestW1 = np.copy(self.W1)
        bestb1 = np.copy(self.b1)
        bestW2 = np.copy(self.W2)
        bestb2 = np.copy(self.b2)
        bestEta = np.copy(self.eta)
        bestLambda = np.copy(self.lambda_reg)

        bestCost = self.computeCost(
            validationSet["data"], validationSet["one_hot"],self.W1, self.b1, self.W2, self.b2)[0]

        for i in range(search_number):
            eta = np.random.uniform(etas[0], etas[1])
            lambda_reg = np.random.uniform(lambdas[0], lambdas[1])
            temp_eta = eta

            print("START: eta: " + str(temp_eta) + " lambda_reg: " +
                  str(lambda_reg))

            self.initialization()
            self.eta = eta
            self.lambda_reg = lambda_reg

            self.fit(X, Y, validationSet=validationSet)

            cost = self.computeCost(
                validationSet["data"], validationSet["one_hot"], self.W1, self.b1, self.W2, self.b2)[0]

            print(" eta: " + str(temp_eta) + " lambda_reg: " + str(lambda_reg) + " cost: " + str(cost))

            if cost < bestCost:
                bestW1 = np.copy(self.W1)
                bestb1 = np.copy(self.b1)
                bestW2 = np.copy(self.W2)
                bestb2 = np.copy(self.b2)
                bestEta = temp_eta
                bestLambda = np.copy(self.lambda_reg)
                bestCost = cost
                print(" Best!")

        print("FINAL BEST => eta: " + str(bestEta) + " lambda_reg: " + str(bestLambda) + " cost: " + str(bestCost))
        self.W1 = np.copy(bestW1)
        self.b1 = np.copy(bestb1)
        self.W2 = np.copy(bestW2)
        self.b2 = np.copy(bestb2)

def main():
    print("Loading dataset...")
    trainSet, validationSet, testSet = loadDataset()
    print("Dataset loaded!")

    
    decay_eta=False
    cyclical_eta=True
    regularization_term=.00021
    batch_size=64
    n_epochs=50
    weight_decay=0.9
    shuffling=True
    hidden_nodes=200
    rho=0.9
    leaky_RELU=True

    class1 = Classifier(decay_eta, cyclical_eta, regularization_term, batch_size, n_epochs, weight_decay, shuffling, hidden_nodes, rho, leaky_RELU)
    class1.fit(trainSet["data"], trainSet["one_hot"], validationSet = validationSet)

    #class1.grid_search(trainSet["data"], trainSet["one_hot"], validationSet, [0.1, 0.01, 0.001, 0.0001, 0.9, 0.00001, 0.5], [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1], [0.95])
    #class1.random_search(trainSet["data"][:, :5000], trainSet["one_hot"][:, :5000], validationSet, 100, [0.001, 0.02], [5.0e-05, .001])

    print("lambda=" + str(regularization_term) + ",", "n epochs=" + str(n_epochs) + ",", "batch size=" + str(batch_size))
    print("Testset final accuracy:")
    print(" " + str(class1.computeAccuracy(testSet["data"], testSet["labels"])))


if __name__ == "__main__":
    main()

