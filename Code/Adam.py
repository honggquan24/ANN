import numpy as np 

class Optimizer_RMSProp:
    # Init the optimizer
    # By default, the learning rate is set to 1.0
    def __init__(self, learning_rate = 0.1, decay = 0., epsilon = 1e-7, rho =
    0.5):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.step = 0
        self.epsilon = epsilon
        self.rho= rho
    # pre update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.step))

    # Update parameters
    def update_params(self, layer):
        if not hasattr(layer, 'weights_cache'):
        # if layers do not contain momentum, create them then fill with zeros

            layer.weights_cache = np.zeros_like(layer.weights)
            layer.biases_cache = np.zeros_like(layer.biases)
            # weights & biases update
            layer.weights_cache = self.rho * layer.weights_cache + (1-self.rho) * layer.dweights**2
            layer.biases_cache = self.rho * layer.biases_cache + (1-self.rho) * layer.dbiases**2
            layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weights_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.biases_cache) + self.epsilon)

    # post update
    def post_update_params(self):
        self.step += 1

class Optimizer_Adam:
# Init the optimizer
# By default, the learning rate is set to 1.0
    def __init__(self, learning_rate = 0.1, decay = 0., epsilon = 1e-7, beta1 =
    0.9, beta2 = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.step = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2= beta2
    # pre update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.step))

    # Update parameters
    def update_params(self, layer):
        if not hasattr(layer, 'weights_cache'):
        # if layers do not contain momentum, create them then fill with zeros

            layer.weights_momentum = np.zeros_like(layer.weights)
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.biases_momentum = np.zeros_like(layer.biases)
            layer.biases_cache = np.zeros_like(layer.biases)

            # Update momentum with current gradient
            layer.weights_momentum = self.beta1 * layer.weights_momentum + (1 - self.beta1) * layer.dweights
            layer.biases_momentum = self.beta1 * layer.biases_momentum + (1 - self.beta1) * layer.dbiases

            # Correct the momentum. step must start with 1 here
            weights_momentum_corrected = layer.weights_momentum / (1 - self.beta1**(self.step +1))
            biases_momentum_corrected = layer.biases_momentum / (1 - self.beta1**(self.step +1))

            # update cache
            layer.weights_cache = self.beta2 * layer.weights_cache + (1-self.beta2) * layer.dweights**2
            layer.biases_cache = self.beta2 * layer.biases_cache + (1-self.beta2) * layer.dbiases**2

            # Obtain the corrected cache
            weights_cache_corrected = layer.weights_cache / (1 - self.beta2**(self.step +1))
            biases_cache_corrected = layer.biases_cache / (1 - self.beta2**(self.step +1))

            # Update weights and biases
            layer.weights += -self.current_learning_rate * weights_momentum_corrected / (np.sqrt(weights_cache_corrected) + self.epsilon)
            layer.biases += -self.current_learning_rate * biases_momentum_corrected/ (np.sqrt(biases_cache_corrected) + self.epsilon)
            # post update
    def post_update_params(self):
        self.step += 1
    # Init the optimizer
    # By default, the learning rate is set to 1.0
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, beta1= 0.9, beta2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay

        self.step = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2= beta2
    # pre update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.step))

    # Update parameters
    def update_params(self, layer):
        if not hasattr(layer, 'weights_cache'):
        # if layers do not contain momentum, create them then fill with zeros
            layer.weights_momentum = np.zeros_like(layer.weights)
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.biases_momentum = np.zeros_like(layer.biases)
            layer.biases_cache = np.zeros_like(layer.biases)

            # Update momentum with current gradient
            layer.weights_momentum = self.beta1 * layer.weights_momentum + (1 -  self.beta1) * layer.dweights
            layer.biases_momentum = self.beta1 * layer.biases_momentum + (1 - self.beta1) * layer.dbiases

            # Correct the momentum. step must start with 1 here
            weights_momentum_corrected = layer.weights_momentum / (1 - self.beta1**(self.step +1))
            biases_momentum_corrected = layer.biases_momentum / (1 - self.beta1**(self.step +1))

            # update cache
            layer.weights_cache = self.beta2 * layer.weights_cache + (1-self.beta2) * layer.dweights**2
            layer.biases_cache = self.beta2 * layer.biases_cache + (1-self.beta2) * layer.dbiases**2

            # Obtain the corrected cache
            weights_cache_corrected = layer.weights_cache / (1 - self.beta2**(self.step +1))

            biases_cache_corrected = layer.biases_cache / (1 - self.beta2**(self.step +1))
            # Update weights and biases
            layer.weights += -self.current_learning_rate * weights_momentum_corrected / (np.sqrt(weights_cache_corrected) + self.epsilon)
            layer.biases += -self.current_learning_rate * biases_momentum_corrected / (np.sqrt(biases_cache_corrected) + self.epsilon)
    # post update
    def post_update_params(self):
        self.step += 1

class Dense_Regularization:
    def __init__(self, n_inputs, n_neurons, weights_regularizer_l1 = 0,
    weights_regularizer_l2 = 0, biases_regularizer_l1 = 0, biases_regularizer_l2 =
    0):
    # Init eights and biases
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        # Set regularization strength
        self.weights_regularizer_l1 = weights_regularizer_l1
        self.weights_regularizer_l2 = weights_regularizer_l2
        self.biases_regularizer_l1 = biases_regularizer_l1
        self.biases_regularizer_l2 = biases_regularizer_l2
        # forward pass
    def forward(self,inputs):
    #calculate outputs
        self.output = np.dot(inputs,self.weights) + self.biases
        self.inputs = inputs

    # backward pass
    def backward(self,dvalues):
    # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        # Gradients on regularization
        # L1 on weights
        if (self.weights_regularizer_l1 > 0):
            dL1 = np.ones_like(self.weights) # return an array filled of 1, same shape as weights array

            dL1[self.weights < 0] = -1
            self.dweights += self.weights_regularizer_l1 * dL1
        # L2 on weights
        if (self.weights_regularizer_l1 > 0):
            self.dweights += 2 * self.weights_regularizer_l2 * self.weights
        # L1 on biases
        if (self.biases_regularizer_l1 > 0):
            dL1 = np.ones_like(self.biases) # return an array filled of 1, same shape as biases array

            dL1[self.biases < 0] = -1
            self.dbiases += self.biases_regularizer_l1 * dL1
        # L2 on biases
        if (self.biases_regularizer_l1 > 0):
            self.dbiases += 2 * self.biases_regularizer_l2 * self.biases
            # Gradients on values
            self.dinputs = np.dot(dvalues,self.weights.T)

class Dense_Dropout:
    def __init__(self, rate):

        self.rate = 1 - rate # invert the rate
    # forward pass
    def forward(self,inputs):
        self.inputs = inputs
        # generate a scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) / self.rate
        # Compute the output value
        self.output = inputs * self.binary_mask
    # backward pass
    def backward(self,dvalues):
        self.dinputs = dvalues * self.binary_mask