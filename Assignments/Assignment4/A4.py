#!/usr/bin/env python3
# author: RuiQu  rqu@kth.se

import numpy as np
import matplotlib.pyplot as plt
from math import ceil

DATA_FILENAME = "./../dataset/goblet_book2.txt"

def read_data():
    book_data = ''
    with open(DATA_FILENAME, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        # book_data += line.replace('\n', '').replace('\t', '')
        book_data += line

    book_char = []

    for i in range(len(book_data)):
        if not(book_data[i] in book_char):
            book_char.append(book_data[i])
    print(len(book_char))

    return book_data, book_char, len(book_char)

def char_to_ind(char, book_char):
    alphabet_size = len(book_char)
    ind = np.zeros((alphabet_size, 1), dtype=int)
    ind[book_char.index(char)] = 1
    return ind.T

def ind_to_char(ind, book_char):
    return book_char[np.argmax(ind)]

class Vanilla_RNN:
    def __init__(self, d, K, char_list):
        self.m = 100
        self.eta = 0.1
        self.seq_length = 25
        self.d = d
        self.K = K
        self.epsilon = 1e-8

        self.nb_epochs = 4

        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))

        self.U = np.zeros((self.m, self.K))
        self.W = np.zeros((self.m, self.m))
        self.V = np.zeros((self.K, self.m))

        self.grad_b = np.zeros((self.m, 1))
        self.grad_c = np.zeros((self.K, 1))

        self.grad_U = np.zeros((self.m, self.K))
        self.grad_W = np.zeros((self.m, self.m))
        self.grad_V = np.zeros((self.K, self.m))

        self.m_b = np.zeros((self.m, 1))
        self.m_c = np.zeros((self.K, 1))

        self.m_U = np.zeros((self.m, self.K))
        self.m_W = np.zeros((self.m, self.m))
        self.m_V = np.zeros((self.K, self.m))

        self.char_list = char_list

        self.h0 = np.zeros((self.m, 1))

        self.initialization()

    def initialization(self):
        mu = 0
        sigma = 0.01

        self.b = np.zeros(self.b.shape)
        self.c = np.zeros(self.c.shape)

        self.U = np.random.normal(mu, sigma, self.U.shape)
        self.W = np.random.normal(mu, sigma, self.W.shape)
        self.V = np.random.normal(mu, sigma, self.V.shape)

    def softmax(self, x):
        r = np.exp(x) / sum(np.exp(x))
        return r

    def forward_pass(self, x, h, b, c, W, U, V):
        ht = h
        H = np.zeros((self.m, x.shape[1]))
        P = np.zeros((self.K, x.shape[1]))
        A = np.zeros((self.m, x.shape[1]))
        for t in range(x.shape[1]):
            a = np.dot(W, ht) + np.dot(U, x[:, [t]]) + b
            ht = np.tanh(a)
            o = np.dot(V, ht) + c
            p = self.softmax(o)
            H[:, [t]] = ht
            P[:, [t]] = p
            A[:, [t]] = a
        return P, H, A

    def computeCost(self, P, Y):
        loss_sum = 0
        for i in range(P.shape[1]):
            p = P[:, [i]]
            y = Y[:, [i]]
            loss_sum += self.cross_entropy(p, y)
        assert(len(loss_sum) == 1)
        return loss_sum

    def cross_entropy(self, p, y):
        l = - np.log(np.dot(y.T, p))
        assert(len(l) == 1)
        return l

    def test_gradient(self, X_chars, Y_chars):
        X = np.zeros((self.d, self.seq_length), dtype=int)
        Y = np.zeros((self.K, self.seq_length), dtype=int)

        for i in range(2):

            self.h0 = np.random.normal(0, 0.01, self.h0.shape)

            for i in range(self.seq_length):
                X[:, i] = char_to_ind(X_chars[i], self.char_list)
                Y[:, i] = char_to_ind(Y_chars[i], self.char_list)

            P, H1, A = self.forward_pass(
                X, self.h0, self.b, self.c, self.W, self.U, self.V)

            H0 = np.zeros((self.m, self.seq_length))
            H0[:, [0]] = self.h0
            H0[:, 1:] = H1[:, :-1]

            self.compute_gradients(P, X, Y, H1, H0, A, self.V, self.W)
            grad_b, grad_c, grad_V, grad_U, grad_W = self.computeGradientsNumSlow(
                X, Y, self.b, self.c, self.W, self.U, self.V)

            print(sum(abs(grad_b - self.grad_b)) /
                  max(1e-4, sum(abs(grad_b)) + sum(abs(self.grad_b))))
            print(sum(abs(grad_c - self.grad_c)) /
                  max(1e-4, sum(abs(grad_c)) + sum(abs(self.grad_c))))

            print(sum(sum(abs(grad_V - self.grad_V))) /
                  max(1e-4, sum(sum(abs(grad_V))) + sum(sum(abs(self.grad_V)))))
            print(sum(sum(abs(grad_U - self.grad_U))) /
                  max(1e-4, sum(sum(abs(grad_U))) + sum(sum(abs(self.grad_U)))))
            print(sum(sum(abs(grad_W - self.grad_W))) /
                  max(1e-4, sum(sum(abs(grad_W))) + sum(sum(abs(self.grad_W)))))

            self.m_b += np.multiply(self.grad_b, self.grad_b)
            self.m_c += np.multiply(self.grad_c, self.grad_c)
            self.m_U += np.multiply(self.grad_U, self.grad_U)
            self.m_W += np.multiply(self.grad_W, self.grad_W)
            self.m_V += np.multiply(self.grad_V, self.grad_V)

            self.b -= np.multiply(self.eta / np.sqrt(self.m_b + self.epsilon), self.grad_b)
            self.c -= np.multiply(self.eta / np.sqrt(self.m_c + self.epsilon), self.grad_c)
            self.U -= np.multiply(self.eta / np.sqrt(self.m_U + self.epsilon), self.grad_U)
            self.W -= np.multiply(self.eta / np.sqrt(self.m_W + self.epsilon), self.grad_W)
            self.V -= np.multiply(self.eta / np.sqrt(self.m_V + self.epsilon), self.grad_V)

            self.h0 = H1[:, [-1]]

    def compute_gradients(self, P, X, Y, H, H0, A, V, W):
        G = -(Y.T - P.T).T

        self.grad_V = np.dot(G, H.T)
        self.grad_c = np.sum(G, axis=-1, keepdims=True)


        dLdh = np.zeros((X.shape[1], self.m))
        dLda = np.zeros((self.m, X.shape[1]))

        dLdh[-1] = np.dot(G.T[-1], V)
        dLda[:,-1] = np.multiply(dLdh[-1].T, (1 - np.multiply(np.tanh(A[:, -1]), np.tanh(A[:, -1]))))

        for t in range(X.shape[1]-2, -1, -1):
            dLdh[t] = np.dot(G.T[t], V) + np.dot(dLda[:, t+1], W)
            dLda[:,t] = np.multiply(dLdh[t].T, (1 - np.multiply(np.tanh(A[:, t]), np.tanh(A[:, t]))))

        self.grad_W = np.dot(dLda, H0.T)
        self.grad_U = np.dot(dLda, X.T)
        self.grad_b = np.sum(dLda, axis=-1, keepdims=True)

        self.grad_b = np.where(self.grad_b<5, self.grad_b, 5)
        self.grad_b = np.where(self.grad_b>-5, self.grad_b, -5)

        self.grad_c = np.where(self.grad_c<5, self.grad_c, 5)
        self.grad_c = np.where(self.grad_c>-5, self.grad_c, -5)

        self.grad_U = np.where(self.grad_U<5, self.grad_U, 5)
        self.grad_U = np.where(self.grad_U>-5, self.grad_U, -5)

        self.grad_V = np.where(self.grad_V<5, self.grad_V, 5)
        self.grad_V = np.where(self.grad_V>-5, self.grad_V, -5)

        self.grad_W = np.where(self.grad_W<5, self.grad_W, 5)
        self.grad_W = np.where(self.grad_W>-5, self.grad_W, -5)

    def synthezise_text(self, x0, h0, n, b, c, W, U, V):
        Y = np.zeros((self.K, n))
        x = x0
        h = h0

        for i in range(n):
            p, h, _ = self.forward_pass(x, h, b, c, W, U, V)
            label = np.random.choice(self.K, p=p[:, 0])

            Y[label][i] = 1
            x = np.zeros(x.shape)
            x[label] = 1

        return Y

    def fit(self, book_data):

        n = len(book_data)
        nb_seq = ceil(float(n-1) / float(self.seq_length))
        smooth_loss = 0
        ite = 0
        losses = []

        for i in range(self.nb_epochs):
            e = 0
            hprev = np.random.normal(0, 0.01, self.h0.shape)

            # if i != 0:
            #     self.eta /= 10

            print("epoch:", i)

            for j in range(nb_seq):

                if j == nb_seq-1:
                    X_chars = book_data[e:n - 2]
                    Y_chars = book_data[e + 1:n - 1]
                    e = n
                else:
                    X_chars = book_data[e:e + self.seq_length]
                    Y_chars = book_data[e + 1:e + self.seq_length + 1]
                    e += self.seq_length

                X = np.zeros((self.d, len(X_chars)), dtype=int)
                Y = np.zeros((self.K, len(X_chars)), dtype=int)

                for i in range(len(X_chars)):
                    X[:, i] = char_to_ind(X_chars[i], self.char_list)
                    Y[:, i] = char_to_ind(Y_chars[i], self.char_list)

                P, H1, A = self.forward_pass(
                    X, hprev, self.b, self.c, self.W, self.U, self.V)

                H0 = np.zeros((self.m, len(X_chars)))
                H0[:, [0]] = self.h0
                H0[:, 1:] = H1[:, :-1]

                self.compute_gradients(P, X, Y, H1, H0, A, self.V, self.W)

                loss = self.computeCost(P, Y)
                if smooth_loss !=0:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                else:
                    smooth_loss = loss

                self.m_b += np.multiply(self.grad_b, self.grad_b)
                self.m_c += np.multiply(self.grad_c, self.grad_c)
                self.m_U += np.multiply(self.grad_U, self.grad_U)
                self.m_W += np.multiply(self.grad_W, self.grad_W)
                self.m_V += np.multiply(self.grad_V, self.grad_V)

                self.b -= np.multiply(self.eta / np.sqrt(self.m_b + self.epsilon), self.grad_b)
                self.c -= np.multiply(self.eta / np.sqrt(self.m_c + self.epsilon), self.grad_c)
                self.U -= np.multiply(self.eta / np.sqrt(self.m_U + self.epsilon), self.grad_U)
                self.W -= np.multiply(self.eta / np.sqrt(self.m_W + self.epsilon), self.grad_W)
                self.V -= np.multiply(self.eta / np.sqrt(self.m_V + self.epsilon), self.grad_V)

                hprev = H1[:, [-1]]

                if ite % 100 == 0:
                    losses.append(smooth_loss)

                if ite % 1000 == 0:
                    print("ite:", ite, "smooth_loss:", smooth_loss)

                if ite % 10000 == 0:
                    Y_temp = self.synthezise_text(X[:, [0]], hprev, 200, self.b, self.c, self.W, self.U, self.V)
                    string = ""
                    for i in range(Y_temp.shape[1]):
                        string += ind_to_char(Y_temp[:, [i]], self.char_list)

                    print(string)

                ite += 1

        np.save("U.npz", self.U)
        np.save("V.npz", self.V)
        np.save("W.npz", self.W)
        np.save("b.npz", self.b)
        np.save("c.npz", self.c)
        np.save("loss.npz", losses)

        Y_temp = self.synthezise_text(char_to_ind("H", self.char_list).T, self.h0, 1000, self.b, self.c, self.W, self.U, self.V)
        string = ""
        for i in range(Y_temp.shape[1]):
            string += ind_to_char(Y_temp[:, [i]], self.char_list)

        print(string)

    def computeGradientsNumSlow(self, X, Y, b, c, W, U, V):
        h = 1e-4

        grad_b = np.zeros((self.m, 1))
        grad_c = np.zeros((self.K, 1))
        grad_U = np.zeros((self.m, self.K))
        grad_W = np.zeros((self.m, self.m))
        grad_V = np.zeros((self.K, self.m))

        print("Computing b gradient")

        for i in range(b.shape[0]):
            b_try = np.copy(b)
            b_try[i] -= h

            P, _, _, = self.forward_pass(X, self.h0, b_try, c, W, U, V)
            c1 = self.computeCost(P, Y)

            b_try = np.copy(b)
            b_try[i] += h

            P, _, _, = self.forward_pass(X, self.h0, b_try, c, W, U, V)
            c2 = self.computeCost(P, Y)
            grad_b[i] = (c2 - c1) / (2 * h)

        print("Computing c gradient")

        for i in range(c.shape[0]):
            c_try = np.copy(c)
            c_try[i] -= h

            P, _, _, = self.forward_pass(X, self.h0, b, c_try, W, U, V)
            c1 = self.computeCost(P, Y)

            c_try = np.copy(c)
            c_try[i] += h

            P, _, _, = self.forward_pass(X, self.h0, b, c_try, W, U, V)
            c2 = self.computeCost(P, Y)
            grad_c[i] = (c2 - c1) / (2 * h)

        print("Computing V gradient")

        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                V_try = np.copy(V)
                V_try[i][j] -= h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W, U, V_try)
                c1 = self.computeCost(P, Y)

                V_try = np.copy(V)
                V_try[i][j] += h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W, U, V_try)
                c2 = self.computeCost(P, Y)
                grad_V[i][j] = (c2 - c1) / (2 * h)

        print("Computing U gradient")

        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                U_try = np.copy(U)
                U_try[i][j] -= h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W, U_try, V)
                c1 = self.computeCost(P, Y)

                U_try = np.copy(U)
                U_try[i][j] += h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W, U_try, V)
                c2 = self.computeCost(P, Y)
                grad_U[i][j] = (c2 - c1) / (2 * h)

        print("Computing W gradient")

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.copy(W)
                W_try[i][j] -= h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W_try, U, V)
                c1 = self.computeCost(P, Y)

                W_try = np.copy(W)
                W_try[i][j] += h

                P, _, _, = self.forward_pass(X, self.h0, b, c, W_try, U, V)
                c2 = self.computeCost(P, Y)
                grad_W[i][j] = (c2 - c1) / (2 * h)

        return grad_b, grad_c, grad_V, grad_U, grad_W

def plot():
    with open("loss.npz.npy") as f:
        loss = list(np.load("loss.npz.npy").reshape(3102, 1))
    print(loss)
    loss_plot = plt.plot(loss, label="training loss")
    plt.xlabel('epoch (divided by 100)')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('graph.png')
    plt.show()

def main():

    book_data, book_char, K = read_data()
    net = Vanilla_RNN(K, K, book_char)
    net.fit(book_data)
    #net.test_gradient(book_data[:net.seq_length], book_data[1:net.seq_length+1])
if __name__ == '__main__':
    main()
    #plot()
