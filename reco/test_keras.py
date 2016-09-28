import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation

from sklearn.metrics import mean_squared_error

tau=2*np.pi

#  Generating data
## Generate X by uniformly sampling the interval [0,tau) 500 times
## Generate targets Y by 3 * sin(x) + 1 + e for error e
## error e defined by e ~ N(0,0.5) (drawn from normal with mean 0, std deviation 0.5

np.random.seed(29)
N = 500
X = np.random.random((N,1))*tau
Y = np.sin(X)*3+1+np.random.normal(0,0.5,(N,1))

fig = plt.plot(np.linspace(0,tau), 3*np.sin(np.linspace(0,tau))+1, 'r')
fig = plt.plot(X, Y, 'b.')
lims = plt.axis([0,tau,-6,6])
#plt.show()

#  Splitting Data
I = np.arange(N)
np.random.shuffle(I)
n = 400

## Training sets
xtr = X[I][:n]
ttr = Y[I][:n]
## Testing sets
xte = X[I][n:]
tte = Y[I][n:]

# Multilayer Perceptron
model = Sequential()    # Feedforward
model.add(Dense(3, input_dim=1))
model.add(Activation('tanh'))
model.add(Dense(1))
model.compile('sgd', 'mse')

hist = model.fit(xtr, ttr, validation_split=0.1, nb_epoch=250)

print type(xtr), xtr.shape
print type(ttr), ttr.shape


pred = model.predict(xte)

plt.plot(xte, pred, 'yo')
plt.show()

print "error:",mean_squared_error(tte, pred)
