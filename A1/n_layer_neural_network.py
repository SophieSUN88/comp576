from typing import *
import numpy as np
import matplotlib.pyplot as plt

from three_layer_neural_network import plot_decision_boundary, generate_data

class DeepNeuralNetwork(object):
    """
    This class builds and trains a deep neural network
    """

    def __init__(self, nn_dims, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_dims: dimension of each layers
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_dims = nn_dims
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        self.W = []
        self.b = []

        np.random.seed(seed)
        for i in range(len(nn_dims)-1):
            self.W.append(np.random.randn(self.nn_dims[i], self.nn_dims[i+1])
                          / np.sqrt(self.nn_dims[i]))
            self.b.append(np.zeros((1, self.nn_dims[i+1])))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE

        if type == "tanh":
            return np.tanh(z)
        elif type == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif type == "relu": # negative -> 0, positive -> same
            return np.maximum(0, z)
        else:
            return None

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE

        # YOU IMPLMENT YOUR actFun HERE
        act_func_dict = {
            "tanh":np.tanh(z),
            'sigmoid':1/ (1+np.exp(-z)),
            "relu":np.maximum(0,z)
        }
        if type in act_func_dict:
            return act_func_dict[type]
        return None

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE

        self.z = []
        self.a = []
        for i in range(len(self.W)):
            if i == 0:
                self.z.append(np.dot(X, self.W[i]) + self.b[i])
            else:
                self.z.append(np.dot(self.a[i-1], self.W[i]) + self.b[i])
            if i != len(self.W) - 1:
                self.a.append(actFun(self.z[i]))
        exp_scores = np.exp(self.z[len(self.z)-1])
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE

        probs = np.exp(self.z[len(self.z)-1]) / \
                np.sum(np.exp(self.z[len(self.z)-1]), axis=1, keepdims=True)
        data_loss_single = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(data_loss_single)

        # Add regulatization term to loss (optional)
        W_sum = 0
        for i in len(self.W):
            W_sum += np.sum(np.square(self.W[i]))
        data_loss += self.reg_lambda / 2 * W_sum
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2, ... dL/dn, dL/bn in two lists
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        delta = self.probs
        delta[range(num_examples), y] -= 1

        dW = []
        db = []
        for i in range(len(self.z)):
            index = len(self.z) - i - 1
            if index != 0:
                dW.insert(0, np.dot(self.a[index - 1].T, delta))
                db.insert(0, np.sum(delta, axis=0, keepdims=True))
                delta = np.dot(delta, self.W[index].T) * \
                        self.diff_actFun(self.z[index-1], type=self.actFun_type)
            else:
                dW.insert(0, np.dot(X.T, delta))
                db.insert(0, np.sum(delta, axis=0, keepdims=False))

        return dW, db

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW, db = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            for i in range(len(dW)):
                # print(dW[i].shape)
                # print(self.W[i].shape)
                dW[i] += self.reg_lambda * self.W[i]

            # Gradient descent parameter update
            for i in range(len(self.W)):
                self.W[i] += -epsilon * dW[i]
                self.b[i] += -epsilon * db[i]

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)

def main():

    # # generate and visualize Make-Moons dataset
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    # model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3 , nn_output_dim=2, actFun_type='tanh')
    model = DeepNeuralNetwork(nn_dims=[2, 3, 2], actFun_type='tanh')
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y)

if __name__ == "__main__":