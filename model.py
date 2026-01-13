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
"Rectified Linear Unit

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
#perfect for back propagation which we'll need to do

rng = np.random.default_rng(42)
D_in = 2
H = 16
D_out = 1

#init for ReLU layer, small init for output
W1 = rng.standard_normal((D_in, H)) * np.sqrt(2.0/D_in)
b1 = np.zeros((H,))
W2 = rng.standard_normal((H, D_out)) * 0.1
b2 = np.zeros((D_out,))

#Training the model
lr = 0.05 #learning rate - scalar multiplier on gradient vector
#small lr means tiny cautious jumps for training and larger lr means bigger jumps
epochs = 2000 #Passes through the data set/how many times we train it

for epoch in range(1, epochs + 1):
    z1 = X_train @ W1 + b1 #(N, H) 
    #matrix multiplies inputs with first layer weights and then adds bias
    #essentially each input point is projected into H hidden features
    a1 = relu(z1) #(N, H)
    z2 = a1 @ W2 + b2 #(N, 1)
    y_prob = sigmoid(z2).reshape(-1) #(N,)

    loss = bce_loss(y_train, y_prob)

    #Bacwkard propagation (manual gradients)
    N = X_train.shape[0]
    dz2 = (y_prob - y_train) / N #(N,)
    dz2 = dz2.reshape(-1, 1)

    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_grad(z1)

    dW1 = X_train.T @ dz1
    db1 = dz1.sum(axis=0)

    #Update here
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    #Periodically evaluate every 200 runs
    if epoch % 200 == 0 or epoch == 1:
        z1v = X_val @ W1 + b1
        a1v = relu(z1v)
        z2v = a1v @ W2 + b2
        yv_prob = sigmoid(z2v).reshape(-1)

        val_loss = bce_loss(y_val, yv_prob)
        train_acc = accuracy(y_train, y_prob)
        val_acc = accuracy(y_val, yv_prob)

        print(f"epoch {epoch:4d} | loss {loss:.4f} | val_loss {val_loss:.4f} | accuracy {train_acc:.3f} | val_acc {val_acc:.3f}")
print("done")