from NeuralNetwork import ArtificialNeuralNetwork


# Net that simulates a OR gate
net1 = ArtificialNeuralNetwork([2, 1, 1])

x_values = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_values = [[0], [1], [1], [1]]

net1.train(x_values, y_values, 90)
print(net1.query([0, 1]))


# Net that simulates an AND gate
net2 = ArtificialNeuralNetwork([2, 1, 1])

x_values = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_values = [[0], [0], [0], [1]]

net2.train(x_values, y_values, 90)
print(net1.query([1, 1]))


# Net that simulates even or odd number machine
net3 = ArtificialNeuralNetwork([1, 50, 1])

x_values = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
y_values = [[1], [0], [1], [0], [1], [0], [1], [0], [1], [0]]

net3.train(x_values, y_values, 90, 0.1)
print(net3.query([7]))


# Net that returns 1 when input is positive and 0 when negative
net4 = ArtificialNeuralNetwork([1, 50, 1])

x_values = [[-3], [-2], [-1], [1], [2], [3]]
y_values = [[0], [0], [0], [1], [1], [1]]

net4.train(x_values, y_values, 10, 0.5)
print(net4.query([3]))