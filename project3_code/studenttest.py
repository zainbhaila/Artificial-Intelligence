import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
#import instrNet
import studNet

data = datasets.make_moons(n_samples=2500, noise=0.1)



X = data[0]
y = np.expand_dims(data[1], 1)



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, )

xPass = x_train.T
yPass = y_train.T
#studentNet = instrNet.test_train(xPass, yPass)
studentNet = studNet.test_train(xPass, yPass)

results = studentNet.predict(x_test.T)
print(accuracy_score(y_test.T[0], results[0]))


data = datasets.make_moons(n_samples=2500, noise=0.2)



X = data[0]
y = np.expand_dims(data[1], 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, )

xPass = x_train.T
yPass = y_train.T

#studentNet = instrNet.test_train(xPass, yPass)
studentNet = studNet.test_train(xPass, yPass)
results = studentNet.predict(x_test.T)
print(accuracy_score(y_test.T[0], results[0]))