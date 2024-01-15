import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

data_train = pd.read_csv("./lab1_dataset/X_train.csv")
data_train_val = data_train.values
print(data_train_val.shape)
label_train = pd.read_csv("./lab1_dataset/Y_train.csv")
label_train_val = label_train.values
print(label_train_val.shape)

data_test = pd.read_csv("./lab1_dataset/X_test.csv")
data_test_val = data_test.values
label_test = pd.read_csv("./lab1_dataset/Y_test.csv")
label_test_val = label_test.values


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(z, y):
    return np.mean(-y.flatten() * np.log(np.clip(z, np.finfo(float).eps, 1.0)) - (
                (1 - y.flatten()) * np.log(np.clip(1 - z, np.finfo(float).eps, 1.0))))


def grad(x, pre, y):
    return np.dot(x.T, pre - y.flatten()) / x.shape[0]


def gradientdescent(x, y):
    #w = np.zeros((x.shape[1]))
    w = np.random.uniform(-0.1, 0.1, size=x.shape[1])
    times = 100000
    rate = 0.001
    losses = []
    for i in range(times):
        z = np.dot(x, w)
        h = sigmoid(z)
        loss1 = loss(h, y)
        rd = np.dot(x.T, (h - y.flatten())) / len(y)
        w -= rate * rd
        if i % 100 == 0:
            losses.append(loss1)

    return w, losses


start = time.time()
w, l_test = gradientdescent(data_train_val, label_train_val)
y_train = sigmoid(np.dot(data_train_val, w))
for i in range(len(y_train)):
    if y_train[i] < 0.5:
        y_train[i] = 0
    else:
        y_train[i] = 1

count1 = 0

for i in range(len(y_train)):
    if label_train_val[i] == y_train[i]:
        count1 = count1 + 1

y_pred = sigmoid(np.dot(data_test_val, w))
for i in range(len(y_pred)):
    if y_pred[i] < 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1

count2 = 0

for i in range(len(y_pred)):
    if label_test_val[i] == y_pred[i]:
        count2 = count2 + 1
print('train accuracy', count1 / len(label_train_val))
print('test accuracy', count2 / len(label_test_val))
print(f'losses is {l_test}')
end = time.time()
plt.plot(l_test, marker='o', linestyle='-')
plt.show()
print("training time cost: {} seconds".format(end - start))
