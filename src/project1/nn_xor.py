# import the necessary packages
from utility.nn import NeuralNetwork
import numpy as np
# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
# define our 2-2-1 neural network and train it
print("[INFO] training 2-2-1 neural network...")
nn = NeuralNetwork.NeuralNetwork([2,2,2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000,displayUpdate=1000)
# now that our 2-2-1 neural network is trained, we can evaluate it
print("[INFO] testing 2-2-1 neural network...")


# now that our network is trained, loop over the XOR data points
for (x, target) in zip(X, y):
    # make a prediction on the data point and display the result
    # to our console
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(
    x, target[0], pred, step))