import random
import math
import numpy as np
from copy import deepcopy

class NeuralNetwork:

  def __init__(self, blueprint):
    self.blueprint = blueprint
    self.weights = []
    self.bias = []
    self.initialize_net()


  def initialize_net(self):
    for i in range(len(self.blueprint)-1):
      self.weights.append([])
      for j in range(self.blueprint[i+1]):
        self.weights[i].append([])
        for k in range(0,self.blueprint[i]):
          self.weights[i][j].append(random.random())

    self.bias = []
    for i in range(len(self.blueprint)-1):
      self.bias.append([n / 100 for n in random.sample(range(1, 100), self.blueprint[i+1])])


  @staticmethod
  def sigmoid(x):
    return round(1/(1+math.exp(-x)), 2)


  @staticmethod
  def inverse_sigmoid(x):
    return round(math.log(x/(1-x)), 2)


  def query(self, input_list):
    if len(input_list) != self.blueprint[0]:
      raise Exception(f"Not enough inputs. You need {self.blueprint[0]} inputs")
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
      return new[-1]


  def activation(self, input_list, weights, bias):
    return self.sigmoid(np.dot(input_list, weights).sum() + bias)


  def loss(self, expected, real):
    pass


  def gradient_discent(self):
    pass


  def train(self, d):
    pass