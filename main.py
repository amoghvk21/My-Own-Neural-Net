from NeuralNetwork import ArtificialNeuralNetwork

x_values = [[1], [2], [3], [4], [5], [6], [7], [8], [9]]
y_values = [[0], [1], [0], [1], [0], [1], [0], [1], [0]]
net = ArtificialNeuralNetwork([1, 3, 3, 1])


print(net.loss(x_values, y_values))