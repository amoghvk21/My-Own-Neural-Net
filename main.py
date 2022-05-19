from NeuralNetwork import ArtificialNeuralNetwork

'''
x_values = [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1]]
y_values = [[0, 1], [1, 1], [0, 1], [1, 1], [0, 1], [1, 1], [0, 1], [1, 1], [0, 1]]
net = ArtificialNeuralNetwork([2, 3, 3, 1])
'''

net = ArtificialNeuralNetwork([1, 30, 1])

x_values = [[1], [2], [3], [4], [5], [6]]
y_values = [[0], [1], [0], [1], [0], [1]]

net.train(x_values, y_values, 40)
print(net.query([99]))