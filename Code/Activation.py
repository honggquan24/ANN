#%%
#Activation Class

import numpy as np
import Loss as loss

#Activation Linear function

class Activation_Linear:
    # Forward Pass
    def forward(self,inputs):
        #Calculate output values from input
        self.output = inputs
    # Backward Pass
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
####################################################################
#Activation function Sigmoid

class Activation_Sigmoid:
    # Forward Pass
    def forward(self,inputs):
        #Calculate output values from input
        self.output = 1 / (1 + np.exp(-inputs))

####################################################################
#Activation function ReLU
class Activation_ReLU:
    #Forward Pass
    def forward(self,inputs):
        #Calculate output values from input
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    # Backward Pass
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        # Put to zero gradient where input is negative
        self.dinputs[self.inputs <= 0] = 0

####################################################################
# Softmax Activation Class Definition
class Activation_SoftMax():

    #Forward pass:
    def forward(self,inputs):

        #Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis = 1,keepdims=True))
        self.exp_values = exp_values

        #Normalized them for each sample
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilities

    # Backward pass:
    def backward(self,dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output,single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten ouput array
            single_output = single_output.reshape(-1,1)
            
            # Calculate the Jacobian Matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

class Activation_Softmax_Loss_CategoricalCrossEntropy():

    # Create activation and loss function object
    def __init__(self):
        self.activation = Activation_SoftMax()
        self.loss = loss.Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)

        # Set the output
        self.output = self.activation.output

        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)

        #if labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis =1)

        # Copy to safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradients
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradients
        self.dinputs = self.dinputs / samples
# %%
