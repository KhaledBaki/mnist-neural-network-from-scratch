# ============================================================
# Khaled Abdul-Baki
# May 17, 2026
# SIMPLE NEURAL NETWORK FROM SCRATCH (USING NUMPY ONLY)
# ============================================================
# SENTIMENT OF THIS PROJECT:
# - This project follows a tutorial I found on the internet specifically from 
#   "Samson Zhang" in YouTube and his video on Neural Networks.
# 
# - With the growth of AI, I wanted to learn how the basic building blocks of it worked,
#   starting with neural networks.
#
# - I spent an entire week reviewing linear algebra as well as watching a large sum of tutorials,
#   and lectures online regarding how such networks function.
#
# MY PERSONAL GOAL BEHIND THIS PROJECT:
# - We used AI everyday, and I wanted to learn how the core building blocks worked since it is a field that 
#   interests me!
#
# This network is designed for the MNIST dataset:
# - 28x28 grayscale images -> 784 inputs
# - output classes are digits 0-9 (10 Digits)
#
# Structure of this Neural Network:
#
# Stage 1: Input Layer (784 neurons i.e each pixel from 28 by 28 image) 
#          
# Stage 2: Hidden Layer (10 neurons, Rectified Linear Unit "ReLU" activation)
#          
# Stage 3: Output Layer (10 neurons, Softmax activation for probabilities)
#
# ============================================================
# RESULTS OF THIS NEURAL NETWORK:
# ============================================================
# - with a learning rate of 0.1 and 1000 iterations, our accuracy was 88.39%
# - Initially the models accuracy increased rapidly, but as we progressed through more itterations,
#   the rate our accuracy got better decreased. A small explanation to why the rate of increase dropped is
#   because at first the model knows of NOTHING so the amount that needed to be changed was HUGE and
#   as we train the model more with more iterations, LESS needs to be altered each time.
#
# ============================================================
# LIMITATIONS OF THIS NEURAL NETWORK:
# ============================================================
# - having only one hidden layer is a big limitation for this network.
# - having only 10 neurons in the hidden layer is also a limitation.
# 
# OVERALL EXPLANATION:
# - if we have more layers and more neurons in the hidden layer, we have more parameterswe can tweak and 
#   change. The more parameters we have, the more room for change we have. Its like having a basic car that 
#   turns roughly v.s a more complex car that turns more smoothly.
#
# HOW TO IMPROVE ACCURACY:
# - increase the number of hidden layers.
# - increase the number of neurons per hidden layer.


# ============================================================
# IMPORTS:
# ============================================================

import numpy as np                      # Used for matrix math / linear algebra
import pandas as pd                     # Used to read CSV datasets
from matplotlib import pyplot as plt    # Used for visualization later


# ============================================================
# DATA ORGANIZATION SECTION:
# ============================================================

# Read dataset from CSV file
data = pd.read_csv('/kaggle/input/competitions/digit-recognizer/train.csv')

# Convert dataframe into a NumPy array
data = np.array(data)

# ============================================================
# MATRIX RELATED METRICS:
# ============================================================
#
# m = number of rows 
# n = number of columns 
m, n = data.shape

# Shuffle the dataset randomly
np.random.shuffle(data)


# ============================================================
# VALIDATION DATA:
# ============================================================
# REASON FOR VALIDATION DATA:
# - Preserving data is used to verify and prevent overfitting after training the model.
# - If we train the model with all the data, it would be overfit and that would destroy the models
#   learning ability to predict/perform well on unseen data.
data_dev = data[0:1000].T

# First row contains the labels
Y_dev = data_dev[0]

# The remaining rows contain the image pixel values 28x28 -> 784
X_dev = data_dev[1:n]

# ============================================================
# NORMALIZING PIXEL VALUES:
# ============================================================
# Normalize pixel values from:
# 0-255 -> 0-1 (Simpler range)
# reducing the range of pixel values helps training.
X_dev = X_dev / 255.


# ============================================================
# TRAINING DATA:
# ============================================================
# Everything apart from the validation data is used to train the model!
data_train = data[1000:m].T

Y_train = data_train[0]

X_train = data_train[1:n]

# Normalizing the pixel value inputs
X_train = X_train / 255.


# ============================================================
# PARAMETER INITIALIZATION
# ============================================================

def init_params():
    # ============================================================
    # SECTION SUMMARY
    # ============================================================
    # FIRST HIDDEN LAYER SUMMARY:
    #
    # - for the first hidden layer, each neuron has a weight and a bias.
    # - the first set of weights are influences by the 784 pixels of the image.
    # - and each neuron in the first layer has a bias "used to contorl the flexibility"
    #   of the neural network
    #
    # OUTPUT LAYER SUMMARY:
    #
    # - for the output layer, each neuron also has a weight and a bias.
    # - the output set of weights are influences by the 10 previous output of the neurons from
    #   the neurons of the first hidden layer.
    # - and each neuron in the output layer has a bias "used to contorl the flexibility"
    #   of the neural network.
    #
    # FINALLY:
    # - we return all the initialized parameters
    
    # ============================================================
    # FIRST HIDDEN LAYER
    # ============================================================
    # FIRST HIDDEN LAYER EXPLANATION:
    # - W1 shape = (10, 784) "First Weight"
    # - 10 neurons in first hidden layer
    # - 784 inputs from image pixels
    W1 = np.random.rand(10, 784) - 0.5

    # b1 shape = (10, 1) "First Bias"
    # One bias per hidden neuron
    b1 = np.random.rand(10, 1) - 0.5
    
    # ============================================================
    # SECOND HIDDEN LAYER:
    # ============================================================
    # SECOND HIDDEN LAYER EXPLANATION:
    # - W2 shape = (10, 10) "Second Weight"
    # - 10 output neurons
    # - receiving 10 hidden activations from hidden layer
    W2 = np.random.rand(10, 10) - 0.5
    
    # b1 shape = (10, 1) "Second Bias"
    # One bias per output neuron
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2


# ============================================================
# ACTIVATION FUNCTIONS:
# ============================================================

    # ============================================================
    # SECTION SUMMARY:
    # ============================================================
    # ReLU ACTIVATION FUNCTION PURPOSE:
    # - Introduces non-linearity so that the network is NOT a simple linear combination.
    # - If the network was to be a linear combination, "Teaching the network" would be 
    #   impractical.
    # - The purpose of ReLU is if the output is negative return 0, else return the value.
    #
    # ReLU DERIVATIVE FUNCTION PURPOSE:
    # - used to enable backpropagation.
    # - unlike sigmoid and tanh functions, the ReLU derivative prevents vanishing gradient values.
    # - only returns 1 or 0, which is efficient computationally for back/forward passes.
    #
    # Softmax FUNCTION PURPOSE:
    # - used in the final layer in a classification model.
    # - normalizes the output function, instead of big and non-comprehendable numbers, we use normalized values.
    #   this means the value is bounded by 0 or 1 always, and it is more or less the probability of correctness.

def ReLU(Z):
    return np.maximum(0, Z)


def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0))

    return expZ / np.sum(expZ, axis=0)


# ============================================================
# FORWARD PROPAGATION:
# ============================================================
# APPLYING FORWARD PROPAGATION:
# - forward propagation is used to calculate the output of each layer of the neural network.
# - it is a prediction of the output.
# - it aids in backpropagation after implementing a loss function, so it all works together as key helpers.
# - simply do the matrix multiplication between the inputs and the weights plus the bias, their outputs IS the next layer.
#

def forward_prop(W1, b1, W2, b2, X):

    # ========================================================
    # LAYER 1 FORWARD PROPAGATION: 
    # ========================================================

    # (FIRST WEIGHT) DOT (FIRST LAYER) + FIRST BIAS
    Z1 = W1.dot(X) + b1

    # Apply FIRST ACTIVATION USING ReLU function to prepare the first hidden layer
    A1 = ReLU(Z1)

    # ========================================================
    # LAYER 2 FORWARD PROPAGATION + SOFTMAX FUNCTION:
    # ========================================================
    
    # (SECOND WEIGHT) DOT (FIRST LAYER) + FIRST BIAS
    Z2 = W2.dot(A1) + b2

    # Convert the outputs to probabilities using the softmax function
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


# ============================================================
# ONE HOT ENCODING:
# ============================================================
# APPLYING ONE HOT ENCODING:
# - instead of using classification by numerical values in an increasing order, we use one hot encoding.
# - this method assignes the classified object to columns, accordingly each one has a specific (1), think like
#   "Identity Matrix".
# - this method is cleaner and simpler and it avoids wrong conclusions.

def one_hot(Y):

    one_hot_Y = np.zeros((Y.size, Y.max() + 1))

    one_hot_Y[np.arange(Y.size), Y] = 1

    return one_hot_Y.T


# ============================================================
# BACKPROPAGATION:
# ============================================================

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # ============================================================
    # OVERALL GRADIENT WEIGHT INTERPRETATION:
    # ============================================================
    # - Positive -> Decrease Weight
    # - Negative -> Increase Weight
    # - Zero     -> Optimal State
    #
    # ============================================================
    # NUMBER OF TRAINING EXAMPLES:
    # ============================================================
    # IMPORTANCE:
    # - the number of training examples helps backpropagation learn.
    #   (Used in later functions). 
    # - just the size of the dataset
    m = Y.size

    # Convert labels to one-hot vectors i.e. (Identity Matrix)
    one_hot_Y = one_hot(Y)

    # ========================================================
    # OUTPUT LAYER GRADIENTS:
    # ========================================================

    # ========================================================
    # MEAN SQUARE ERROR:
    # ========================================================
    # EXPLANATION:
    # prediction - actual
    dZ2 = A2 - one_hot_Y

    # Gradient of output weights:
    # How much should output weights change.
    dW2 = (1 / m) * dZ2.dot(A1.T)

    # Output bias gradients:
    # averages out all the output errors
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # ========================================================
    # HIDDEN LAYER GRADIENTS:
    # ========================================================

    # Backpropagate error in the hidden layer
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)

    # Hidden weight gradients
    dW1 = (1 / m) * dZ1.dot(X.T)

    # Average hidden layer error across all the samples
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


# ============================================================
# UPDATE PARAMETERS:
# ============================================================

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):

    # General Gradient Descent Formula:
    # W_NEW = W_OLD - learning_rate * gradient
    
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    
    #return the updated parameters
    return W1, b1, W2, b2


# ============================================================
# PREDICTIONS FOR THE OUTPUT:
# ============================================================

def get_predictions(A2):
    
    #just returns the maximum from all the arguments passed
    return np.argmax(A2, axis=0)


# ============================================================
# ACCURACY:
# ============================================================

def get_accuracy(predictions, Y):

    print("Predictions:", predictions)
    print("Actual Labels:", Y)

    return np.sum(predictions == Y) / Y.size


# ============================================================
# TRAINING FOR THE MODEL LOOP:
# ============================================================

def gradient_descent(X, Y, alpha, iterations):

    # Initializing the weights and biases of the neurons
    W1, b1, W2, b2 = init_params()

    # Training loop for the model
    for i in range(iterations):

        # ====================================================
        # FORWARD PROPAGATION:
        # ====================================================

        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        # ====================================================
        # BACKPROPAGATION:
        # ====================================================

        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

        # ====================================================
        # UPDATING THE PARAMETERS:
        # ====================================================

        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # ====================================================
        # PRINTING THE PROGRESS OF MODEL TRAINING:
        # ====================================================

        if i % 10 == 0:

            print("Iteration:", i)

            predictions = get_predictions(A2)

            accuracy = get_accuracy(predictions, Y)

            print("Accuracy:", accuracy)
            print("---------------------------")

    return W1, b1, W2, b2


# ============================================================
# ACTUALLY TRAIN MODEL:
# ============================================================

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 1000)