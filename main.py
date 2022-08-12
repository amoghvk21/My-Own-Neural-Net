from neuralnet import ArtificialNeuralNetwork


n = ArtificialNeuralNetwork([2, 10, 2])

n.back_prop([[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 1], [0, 1], [0, 1], [1, 0]], 500, 0.5)
print(n.forward_prop([[1, 1]]))