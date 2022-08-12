import numpy as np


class NeuralNetwork:
    '''
    2 -> 3 -> 2 network
    '''
    
    def __init__(self):
        self.w1 = np.random.random((3, 2))
        self.w2 = np.random.random((2, 3))

        self.b1 = np.random.random((3, 1))
        self.b2 = np.random.random((2, 1))

        self.z1 = None # Before activation function
        self.z2 = None

        self.a0 = None # After activation function
        self.a1 = None
        self.a2 = None


    def forward_prop(self, x):
        self.a0 = np.array(x).T
        self.z1 = np.add(np.dot(self.w1, self.a0), self.b1)
        self.a1 = self.ReLU(self.z1)

        self.z2 = np.add(np.dot(self.w2, self.a1), self.b2)
        self.a2 = self.softmax(self.z2)

        return self.a2


    @staticmethod
    def ReLU(x):
        ans = []
        for y in x:
            for z in y:
                if z < 0:
                    ans.append(0)
                else:
                    ans.append(z)
        return np.array(ans).reshape(x.shape)


    @staticmethod
    def softmax(x):
        x = x.T
        ans = []
        for m in x:
            sum = 0
            for y in m:
                sum += np.exp(y)
            for y in m:
                ans.append(np.exp(y)/sum)

        ans = np.reshape(ans, x.shape)
        return ans.T

    
    def back_prop(self, x, y, epoch=10, l=0.1):

        # Check if x and y are valid lists
        if ((len(x) != len(y)) or (len(x[0]) != 2) or (len(y[0]) != 2)):
            raise Exception("x or y invalid input")

        # Convert y into a matrix of correct form and dimentions
        xn = np.array(x).T
        y = np.array(y).T

        # Print origional loss 
        print(f'accuracy before traning: {self.accuracy(x, y.T)}')
        
        # Iterate epoch times
        for e in range(1, epoch+1):

            # Run forward prop with x so that you get the output layer
            self.forward_prop(x)

            # Equations
            m = y.shape[1]

            self.dz2 = np.subtract(self.a2, y)
            self.dw2 = np.multiply(1/m, np.dot(self.dz2, self.a1.T))
            self.db2 = np.multiply(1/m, np.sum(self.dz2))

            self.dz1 = np.multiply(np.dot(self.w2.T, self.dz2), self.deriv_ReLU(self.z1))
            self.dw1 = np.multiply(1/m, np.dot(self.dz1, xn.T))
            self.db1 = np.multiply(1/m, np.sum(self.dz1))

            # Update the weights and biases accordingly
            self.w1 = np.subtract(self.w1, (l * self.dw1))
            self.w2 = np.subtract(self.w2, (l * self.dw2))
            self.b1 = np.subtract(self.b1, (l * self.db1))
            self.b2 = np.subtract(self.b2, (l * self.db2))

            # Print loss after this epoch
            print(f'epoch {e} done. Accurracy: {self.accuracy(x, y)}. Loss: {self.loss(x, y)}')


    def accuracy(self, x, y):
        return 0


    def loss(self, x, y):
        return 0
        

    @staticmethod
    def deriv_ReLU(x):
        return x > 0