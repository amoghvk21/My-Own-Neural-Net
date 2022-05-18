import random
import math
import numpy as np
from copy import deepcopy

class ArtificialNeuralNetwork:
  '''
  Implimentation of an artifical neural network by Amogh
  '''
  def __init__(self, blueprint):
    self.blueprint = blueprint
    self.weights = []
    self.bias = []
    self.weights, self.bias = self.create_weights_bias()


  def create_weights_bias(self):
    '''
    - Function that returns a new bias and weights matrices in the correct format and structure.
    - Assigns random real numbers from 0 to 1
    '''
    weights = []
    for i in range(len(self.blueprint)-1):
      weights.append([])
      for j in range(self.blueprint[i+1]):
        weights[i].append([])
        for k in range(0,self.blueprint[i]):
          weights[i][j].append(random.random())

    bias = []
    for i in range(len(self.blueprint)-1):
      bias.append([n / 100 for n in random.sample(range(1, 100), self.blueprint[i+1])])

    return weights, bias


  @staticmethod
  def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return (i, x.index(v))


  @staticmethod
  def sigmoid(x):
    '''
    Math sigmoid function used to break the linearity of the artifical neural network
    '''
    return round(1/(1+math.exp(-x)), 2)


  @staticmethod
  def inverse_sigmoid(x):
    '''
    Math inverse sigmoid function used to break the linearity of the artifical neural network
    '''
    return round(math.log(x/(1-x)), 2)


  def query(self, input_list):
    '''
    Accepts a list of inputs and will propogate through the entire neural network and return a vector of the output layer
    '''
    if len(input_list) != self.blueprint[0]:
      raise Exception(f"Incorrect amount of inputs. You need {self.blueprint[0]} inputs")
    else:
      temp = deepcopy(input_list)
      new = []
      i = 0
      for weights in self.weights:
        j = 0
        new.append([])
        for weight in weights:
          new[i].append(self.activation(temp, weight, self.bias[i][j]))
          j += 1
        temp = deepcopy(new[-1])
        i += 1
      self.new = new # debugging
      return new[-1]


  def activation(self, input_list, weights, bias):
    '''
    The activation function for all the nodes in the neural network.
    It accepts the vector of all the inputs and their corresponding weights going into the nodes as well as the bias of that node
    and computes the dot product, adds the bias and feeds it into the sigmoid function to break linearity
    '''
    return self.sigmoid(np.dot(input_list, weights).sum() + bias)


  def loss(self, x_inputs, y_expected):
    '''
    Will accept a list of x_inputs, y_expected. Return a value of how acurate the neural network is at predicting the given data
    - Difference from the predicted value and the actual value
    - Square each answer
    - Sum the vector
    - Iterates through whole dataset
    '''
    sum = 0
    for x, y in zip(x_inputs, y_expected):
      sum += (np.power(np.subtract(self.query(x), y), 2).sum())/len(x)
    return sum

  '''
  def accuracy(self, x_inputs, y_expected):
    Return a percentage of how accurate the model was at predicting the x_inputs and how close it was to the y_expected
    This value not used by the model. More used for the UI for the user as it is easier to understand
    total = len(x_inputs)
    wrong = 0
    for x, y in zip(x_inputs, y_expected):
      if self.
  '''


  def update_bias(self, b):
    '''
    Update the bias
    Need to add more parameters...
    Need to impliment
    '''
    return b


  def update_weights(self, w):
    '''
    Update the weights
    Need to add more parameters...
    Need to impliment
    '''
    return w


  def train(self, x_values, y_values, epoch, l=0.5):
    '''
    Function that is ran by the user to train the network given the:
    - x_values of the dataset
    - y_values of the dataset
    - learning rate: how quickly will the weights and biases change to minimise the cost (default value of 0.5)
    - New weight = old weight + learnrate*(targ-pred)*inputs
    '''
    loss = self.loss(x_values, y_values)
    new_weights = []
    new_bias = []

    i = 0
    while i < epoch:
      i += 1
      for x, y in zip(x_values, y_values):
        pred = self.query(x)
        for w1, b1 in zip(self.weights, self.bias):
          new_bias.append([])
          new_weights.append([])
          for w2, b2 in zip(w1, b1):

            # b2 is the bias for a neuron
            b2n = self.update_bias(b2)
            new_bias[self.index_2d(self.bias, b2)[0]].append(b2n)

            # w2 is a list of all the weights going into a neuron 
            w2n = self.update_weights(w2)
            new_weights[self.index_2d(self.weights, w2)[0]].append(w2n)

      print(f"Epoch {i} Complete. Loss: {self.loss(x_values, y_values)}. Accuracy: {None}")