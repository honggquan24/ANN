#%%
# Class Loss

import numpy as np
import math

# Loss Class
class Loss:
    # Calculate the data and regulazation losses given
    # the output and the ground truth values
    def calculate(self,output,y):
        #Calculate sample losses:
        sample_losses = self.forward(output,y)
        #Calculate mean loss:
        data_loss = np.mean(sample_losses)
        #Return loss:
        return data_loss
    

# Cross Entropy Loss Class
class Loss_CategoricalCrossentropy(Loss): 

    # Forward Pass
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # Clip both sides to not drag mean toward any values
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7) #1e-7 is added to avoid division by 0

        # Probabilities for target values only if categorical labels

        if len(y_true.shape) == 1: # The label array is 1D [0 0 ... 1 ... 0 0]
            correct_confidence = y_pred_clipped[
                range(samples),
                y_true
            ]    # Extract the correct confidence in y_pred by referencing to y_true vector
        elif len(y_true.shape) == 2: # The label array is 2D (vector form: [[0 0 0 ... 1 0 0 ... 0],[0 0 0 ... 1 0 0 ... 0],[0 0 0 ... 1 0 0 ... 0]])
            correct_confidence = np.sum(
                y_pred_clipped*y_true,
                axis = 1
            )

        # Losses calculation
        negative_log_likelihoods = -np.log(correct_confidence)
        return negative_log_likelihoods

# %%
