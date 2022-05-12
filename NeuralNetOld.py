import numpy as np
import math
import random

DEFAULT = 2

def sigmoid(x):
    return round(1/(1+math.exp(-x)), 2)

def inverse_sigmoid(x):
    return round(math.log(x/(1-x)), 2)


class NeuralNetwork:
    def __init__(self, no_of_inputs, no_of_hidden_nodes, no_of_outputs):
        self.no_of_inputs = no_of_inputs
        self.no_of_outputs = no_of_outputs
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.weights = [[DEFAULT] * self.no_of_inputs, [DEFAULT] * self.no_of_outputs] # 2D List of random numbers of length for the set of weights
        self.bias = [DEFAULT, DEFAULT] # Random number
        self.activations = [DEFAULT] * no_of_hidden_nodes

    def getOutput(self, inputs):
        if type(inputs) != list:
            raise Exception("Input needs to be of type list")

        if len(inputs) != self.no_of_inputs:
            raise Exception(f"Input needs to be of length {self.no_of_hidden_nodes}")

        temp = 0

        for x in range(self.no_of_hidden_nodes):
            for y in range(self.no_of_inputs):
                temp += (self.weights[0][y] * inputs[y])
            temp += self.bias[0]
            

        for x in range(self.no_of_hidden_nodes):
            self.activations[x] = sigmoid(np.dot(inputs, self.weights[0][x]) + self.bias[0])

        output = [None] * self.no_of_outputs

        for x in range(self.no_of_outputs):
            output[x] = sigmoid(np.dot(self.activations, self.weights[1][x]) + self.bias[1])
        
        return output