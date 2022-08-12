import numpy as np


class ArtificialNeuralNetwork:
    '''
    Network of any number of layers and number of nodes in each layer
    '''
    
    def __init__(self, blueprint=[2, 3, 2]):
        self.blueprint = blueprint

        self.w = [None]
        self.b = [None]
        for i in range(0, len(blueprint)-1):
            self.w.append(np.random.random((blueprint[i+1], blueprint[i])))
            self.b.append(np.random.random((blueprint[i+1], 1)))

        self.z = [None] * len(blueprint)        # Before activation function
        self.a = [None] * len(blueprint)        # After activation function

        self.dz = [None] * len(blueprint)       # How much to change the raw number for that layer by
        self.dw = [None] * len(blueprint)       # How much to change the weights for that layer by
        self.db = [None] * len(blueprint)       # How much to change the biases for that layer by
        

    
    def accuracy(self, x, y):
        return 0


    def loss(self, x, y):
        return 0
        

    @staticmethod
    def deriv_ReLU(x):
        return x > 0


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


    def forward_prop(self, x):
        self.a[0] = np.array(x).T

        for i in range(1, len(self.blueprint)-1):
            self.z[i] = np.add(np.dot(self.w[i], self.a[i-1]), self.b[i])
            self.a[i] = self.ReLU(self.z[i])

        self.z[len(self.blueprint)-1] = np.add(np.dot(self.w[len(self.blueprint)-1], self.a[len(self.blueprint)-2]), self.b[len(self.blueprint)-1])
        self.a[len(self.blueprint)-1] = self.softmax(self.z[len(self.blueprint)-1])

        return self.a[len(self.blueprint)-1]


    def back_prop(self, x, y, epoch=10, l=0.1):
        
        # Check if x and y are the same length
        if (len(x) != len(y)):
            raise Exception("length of x not the same as length of y")

        # Checl if x and y have correct number of data consistent with the blueprint
        for i, j in zip(x, y):
            if ((len(i) != self.blueprint[0]) or (len(j) != self.blueprint[-1])):
                raise Exception("check the data matches with the input/output layer of blueprint")

        # Convert y into a matrix of correct form and dimentions
        xn = np.array(x).T
        y = np.array(y).T

        # Print origional loss 
        print(f'accuracy before traning: {self.accuracy(x, y.T)}. loss before training: {self.loss(x, y.T)}')
        
        # Iterate epoch times
        for e in range(1, epoch+1):

            # Run forward prop with x so that you get the output layer
            self.forward_prop(x)

            # Equations
            m = y.shape[1]

            # Output layer
            self.dz[len(self.blueprint)-1] = np.subtract(self.a[len(self.blueprint)-1], y)
            self.dw[len(self.blueprint)-1] = np.multiply(1/m, np.dot(self.dz[len(self.blueprint)-1], self.a[len(self.blueprint)-2].T))
            self.db[len(self.blueprint)-1] = np.multiply(1/m, np.sum(self.dz[len(self.blueprint)-1], 1))
            self.db[len(self.blueprint)-1] = np.reshape(self.db[len(self.blueprint)-1], (self.blueprint[-1], 1))

            # Rest
            for i in range(len(self.blueprint)-2, 1, -1):
                self.dz[i] = np.dot(self.w[i+1].T, self.dz[i+1]) * self.deriv_ReLU(self.z[i])
                self.dw[i] = 1/m * np.dot(self.dz[i], self.z[i-1].T) # or self.a
                self.db[i] = np.multiply(1/m, np.sum(self.dz[i], 1))
                self.db[i] = np.reshape(self.db[i], (self.blueprint[i], 1))

            # First layer (not input)
            self.dz[1] = np.dot(self.w[2].T, self.dz[2]) * self.deriv_ReLU(self.z[1])
            self.dw[1] = 1/m * np.dot(self.dz[1], xn.T)
            self.db[1] = np.multiply(1/m, np.sum(self.dz[1], 1))
            self.db[1] = np.reshape(self.db[1], (self.blueprint[1], 1))

            # Update the weights and biases accordingly
            for i in range(1, len(self.blueprint)):
                self.w[i] = np.subtract(self.w[i], np.dot(l, self.dw[i]))
                self.b[i] = np.subtract(self.b[i], (l * self.db[i]))

            # Print loss after this epoch
            print(f'epoch {e} done. Accurracy: {self.accuracy(x, y)}. Loss: {self.loss(x, y)}')