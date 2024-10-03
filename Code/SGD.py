#%%
#Stochastic Gradient Descent with fixe learning rate

import numpy as np

class Optimizer_SGD:

    # Init the optimizer
    # By default, the learning rate is set to 1.0
    
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


class Optimizer_SGD_Decay_Momentum:

    # Init the optimizer
    # By default, the learning rate is set to 1.0
    
    def __init__(self, learning_rate, decay, momentum):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.step = 0
        self.momentum = momentum

    # pre update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.step))


    # Update parameters
    def update_params(self, layer):
        if self.momentum:  # if we use momentum
            if not hasattr(layer, 'weights_momentums'):
                # if layers does not contain momentum, create them then fill with zeros
                layer.weights_momentums = np.zeros_like(layer.weights)
                layer.biases_momentums = np.zeros_like(layer.biases)

            # weights update
            weights_updates = self.momentum * layer.weights_momentums - self.current_learning_rate *layer.dweights 
            layer.weights_momentums = weights_updates
            # biaises update
            biases_updates = self.momentum * layer.biases_momentums - self.current_learning_rate *layer.dbiases
            layer.biases_momentums = biases_updates
        else:  # not using momentum
            weights_updates = -self.current_learning_rate * layer.dweights
            biases_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weights_updates
        layer.biases += biases_updates

    # post update
    def post_update_params(self):
        self.step += 1
# %%
