import time, gc
from Brainiac import Brain
import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = 2 * inputs[: , : 1] - 3 * inputs[:, 1: ]
print("\nthe targets are:\n")
print(targets)

brain = Brain(0.1, 1e-3, np.shape(inputs)[1], 20, np.shape(targets)[1])

print("\ntraining the network...")
t = time.time()
sseTrend = brain.train(inputs, targets, batchSize=3, iterations=5000)
print("\ntraining took: {}s\n".format(time.time() - t))

print("\nfor the inputs:\n")
print(inputs)

print("\nmy predictions are:\n")
t = time.time()
print(brain.predict(inputs))
print("\nprediction took: {}s\n".format(time.time() - t))

# print("\nmy architecture:")
# brain.show()

del inputs, targets, brain, t
gc.collect()