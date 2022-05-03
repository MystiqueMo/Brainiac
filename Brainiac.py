import numpy as np
import random
from tqdm import tqdm
from functools import lru_cache

class Neuron:
    def __init__(self, dimOfInputs, learningRate, reluScale):
        self.weights = np.random.normal(0.0, 100.0, (1 + dimOfInputs, 1))
        self.learningRate = learningRate
        self.reluScale = reluScale

    def __str__(self):
        return "{" + "{}: {}".format(self.weights[0], self.weights[1:].flatten()) + "}"

    # this takes in an arbitrary batch of inputs and targets as tuples and adjusts the neuron weight accordingly...
    def learn(self, inputs, targets=None, errors=None, indx=None, container=None):
        if not(isinstance(targets, type(None)) ^ isinstance(errors, type(None))):  return
        adjustedInputs = np.insert(inputs, 0, 1, axis=1)
        
        if not(isinstance(targets, type(None))):    errors = (targets - adjustedInputs.dot(self.weights))
        
        container[0][: , indx: indx + 1] = errors
        container[1][indx: indx + 1, : ] = self.weights[1:].transpose() / np.linalg.norm(self.weights[1:].transpose())
        
        self.weights = np.add(self.weights, self.learningRate * adjustedInputs.transpose().dot(errors))

    # this takes in an input sample as a tuple and spits out a single number...
    def predict(self, sample):
        self.activation = np.insert(sample, 0, 1).dot(self.weights)[0]
        return self.reluScale * self.activation if self.activation > 0 else 0

class Brain:
    def __init__(self, learningRate, reluScale, *args):
        self.feedForward.cache_clear()
        self.learningRate = learningRate
        self.layers = list()
        [self.layers.append([Neuron(args[i - 1], learningRate, reluScale) for _ in range(args[i])]) for i in range(1, len(args))]
            
    @lru_cache(maxsize=None)
    def feedForward(self, sample, hiddenLayerIndex):
        layer_outputs = [neuron.predict(self.feedForward(sample, hiddenLayerIndex - 1) if hiddenLayerIndex > 0 else sample) for neuron in self.layers[hiddenLayerIndex]]
        
        if hiddenLayerIndex == len(self.layers) - 1:
            layer_outputs = [neuron.activation for neuron in self.layers[hiddenLayerIndex]]
        return tuple(layer_outputs)

    # this takes in a collection of input samples as a tuple and spits out a collection of outputs also as a tuple...
    def predict(self, batchOfInputs, hiddenLayerIndex=None):
        if isinstance(hiddenLayerIndex, type(None)):    hiddenLayerIndex = len(self.layers) - 1
        isFirstRow = True
        for row in batchOfInputs:
            predictions = np.array([self.feedForward(tuple(row), hiddenLayerIndex)]) if isFirstRow else np.concatenate((predictions, np.array([self.feedForward(tuple(row), hiddenLayerIndex)])), axis=0)
            isFirstRow = False
        return predictions

    def backPropagate(self, sampleBatch, layerIndex, targets=None, errors=None):
        if not(isinstance(targets, type(None)) ^ isinstance(errors, type(None))):  return

        layerSampleBatch = self.predict(sampleBatch, hiddenLayerIndex=layerIndex - 1) if layerIndex > 0 else sampleBatch
        
        this_layer_errors = np.zeros((len(layerSampleBatch), len(self.layers[layerIndex])))
        this_layer_weights = np.zeros((len(self.layers[layerIndex]), len(self.layers[layerIndex - 1]) if layerIndex > 0 else np.shape(sampleBatch)[1]))
        
        if not(isinstance(targets, type(None))):
            [neuron.learn(layerSampleBatch, targets=targets[: , j: j + 1], indx=j, container=(this_layer_errors, this_layer_weights)) for j, neuron in zip(range(len(self.layers[layerIndex])), self.layers[layerIndex])]
        else:   [neuron.learn(layerSampleBatch, errors=errors[: , j: j + 1], indx=j, container=(this_layer_errors, this_layer_weights)) for j, neuron in zip(range(len(self.layers[layerIndex])), self.layers[layerIndex])]
        
        if layerIndex == len(self.layers) - 1:  self.sumOfSquaredErrors = np.sum(np.square(this_layer_errors))

        if layerIndex > 0:
            preceding_layer_errors = this_layer_errors.dot(this_layer_weights)
            self.backPropagate(sampleBatch, layerIndex - 1, errors=preceding_layer_errors)

    def learn(self, sampleBatch, targets):
        self.backPropagate(sampleBatch, len(self.layers) - 1, targets=targets)

    # this takes in the whole training data as ndarrays and feeds it in random batches to the learn method as tuples...
    def train(self, inputs, targets, batchSize=1, iterations=None):
        if not isinstance(iterations, type(None)):
            plotFrame = []
            meanSSE = 0
            for _ in tqdm(range(iterations)):
                i = random.randint(0, len(inputs) - batchSize)
                self.learn(inputs[i: i + batchSize, : ], targets=targets[i: i + batchSize, : ])

                meanSSE += self.learningRate * (self.sumOfSquaredErrors - meanSSE)
                plotFrame.append([meanSSE, ])
                
            return plotFrame
        else:
            while True:
                i = random.randint(0, len(inputs) - batchSize)
                self.learn(inputs[i: i + batchSize, : ], targets=targets[i: i + batchSize, : ])

    def show(self):
        [print(f"\nlayer[{self.layers.index(layer) + 1}].neuron[{layer.index(neuron) + 1}] = {neuron}") for layer in self.layers for neuron in layer]
