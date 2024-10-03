#%%
# Dense Layer - Class definition

import numpy as np

class Dense:

    def __init__(self, n_inputs, n_neurons):
        #init eights and biases
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        #calculate outputs
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) +self.biases
           
    # backward pass
    def backward(self,dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        # Gradients on values
        self.dinputs = np.dot(dvalues,self.weights.T)
# %%
