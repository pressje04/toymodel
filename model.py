import numpy as np 

"""
This function generates a non-linear dataset which is great for observing
nueral networks. What it's doing conceptually is generating two interleaving half-circles
or moons. 

@param n 
    The number of data points
@param noise
    randomness added to points, if noise is 0.0 we get perfectly drawn moons
    but noise simulates real world imperfections w/ data like measurement error or
    ambiguity
@param seed
    random number generater seed
    * Same seed every time ==> same dataset every time
    * Different seed ==> different random draw of data

Mentally, let's say you try to draw moons yourself. 
n - number of dots you draw
noise - how shaky your hand is
seed - random pattern of shakiness
"""
def make_moons(n=800, noise=0.15, seed=0):
    rng = np.random.default_rng(seed) #rng = random number generator
    #Split data into 2 classes (moon 1 and moon 2)
    n1 = n // 2
    n2 = n - n1

    #Generates angles for FIRST moon, like sampling points along a half-circle
    t1 = rng.uniform(0, np.pi, n1)

    #convert angles to 2D points.
    #This is classing parametric circle math (x = cos(t), y = sin(t))
    x1 = np.c_[np.cos(t1), np.sin(t1)] + noise * rng.standard_normal((n1, 2)) 

    #Do the same for the second moon now
    t2 = rng.uniform(0, np.pi, n2)
    x2 = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)] + noise * rng.standard_normal((n2, 2))
    #np.c - stacks columns
    #before factoring in noise, you have a perfect half circle

    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])

    #shuffle
    idx = rng.permutation(n)
    return X[idx], y[idx]

X, y = make_moons()

#train/val spit
split = int(0.8 * len(X))
X_train = X[:split]
y_train = y[:split]

X_val = X[split:]
y_val = y[split:]

#Math Helper Functions

#Sigmoid - S shape curve
"""
Turns each score into a probability

* big positive z ==> probability near 1
* big negative z ==> probability near 0
* z = 0 ==> probability 0.5

@param z
    numerical score to be converted
"""
def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

"""
"Rectified Linear Unity

Introduces nonlinearity to the network and prevents it from collapsing
into a single linear transformation

Each neuron:

* outputs 0 when “inactive”
* passes positive signal when “active”

This makes the model behave differently in different regions of input space.

"""
def relu(z):
    return np.maximum(0, z)

"""
Rectified Linear Unit Gradient - needed for backpropigation bc it needs derivatives of ReLU

* Either 1 if z is positive or 0 if it's negative

If the neuron is negative/off, its gradient is 0
If the neuron is positive/on, the gradient passes normally
"""
def relu_grad(z):
    return (z > 0).astype(z.dtype)

"""
Essentially how wrong were my probabilities
This uses binary-cross entropy to measure how good the predicted 
probabilities are (on the return line is the equation for it)

If y=1, the loss is -log(ŷ) ==> wants ŷ → 1
If y=0, the loss is -log(1-ŷ) ==> wants ŷ → 0

This kind of check treats predictions that are confident and wrong 
drastically. 

Ex: if true y = 1:
* Predict ŷ = 0.99, this is slightly off and not a big deal
* Predict ŷ = 0.01  this is a huge loss and problematic
"""
def bce_loss(y_true, y_pred, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

def accuracy(y_true, y_prob):
    y_hat = (y_prob >= 0.5).astype(int)
    return (y_hat == y_true).mean()



# Model: 2-layer MLP (Multi-layer Perceptron) (2 -> hidden -> 1)
# input layer, one or more hidden layers, and an output layer
