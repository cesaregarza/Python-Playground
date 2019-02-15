import math
import random
class Neuron:
    def __init__(self, name, parent):
        self._name = name
        self._parent = parent
        self._weights = {}
        self._bias = 0
        self._outputList = []
        self._inputs = {}
    
    def sigmoid(self, i):
        z = math.exp(i * -1) + 1
        return 1 / z
    
    def calculate(self):
        runningTotal = 0
        for key in inputs:
            value = inputs[key]
            runningTotal += self._weights[key] * value
        
        runningTotal += self._bias
        output = self.sigmoid(runningTotal)

        for i in self._outputList:
            

    
