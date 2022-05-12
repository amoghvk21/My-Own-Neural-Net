import random
import math

class NeuralNetwork:
  def __init__(self, blueprint):
    self.blueprint = blueprint
    self.initialize_net()

  def initialize_net(self):
    pass
    
  def sigmoid(x):
    return round(1/(1+math.exp(-x)), 2)

  def inverse_sigmoid(x):
    return round(math.log(x/(1-x)), 2)

  def query(self):
    pass

  def activation(self):
    pass

  def cost(self, expected, real):
    pass

  def back_propogate(self):
    pass

  def train(self, d):
    pass