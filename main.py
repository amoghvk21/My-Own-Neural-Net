from neuralnet import ArtificialNeuralNetwork


n = ArtificialNeuralNetwork([2, 10, 2])

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0, 1], [0, 1], [0, 1], [1, 0]]

n.back_prop(X, Y, 100, 0.1)
print(n.forward_prop([[1, 1]]))