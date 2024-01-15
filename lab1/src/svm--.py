import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

data_train = pd.read_csv("./lab1_dataset/X_train.csv")
data_train_val = data_train.values
label_train = pd.read_csv("./lab1_dataset/Y_train.csv")
label_train_val = label_train.values
label_train_val[label_train_val == 0] = -1
# print(label_train_val)

data_test = pd.read_csv("./lab1_dataset/X_test.csv")
data_test_val = data_test.values
label_test = pd.read_csv("./lab1_dataset/Y_test.csv")
label_test_val = label_test.values
label_test_val[label_test_val == 0] = -1
print(len(label_test_val))


class SVM:

    def __init__(self, learning_rate=0.000005, lambda_param=0.0001, n_iters=7500):
        self.lr = learning_rate  # 𝛼 in formula
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.shape = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.shape = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            loss = 0
            for idx, x_i in enumerate(X):
                loss += self.update(x_i, y[idx])
            loss /= self.shape[0]
            if _ % 100 == 0:
                self.losses.append(loss)

    def update(self, x, y):
        distance = 1 - (y * (np.dot(x, self.w) + self.b))
        hinge_loss = max(0, distance)
        if hinge_loss == 0:
            self.w -= self.lr * (2 * self.lambda_param * self.w)
        else:
            self.w -= self.lr * (2 * self.lambda_param * self.w - x * np.tile(y, len(x)))
            self.b += self.lr * y
        return hinge_loss

    def predict(self, X):
        eq = np.dot(X, self.w) + self.b
        return np.sign(eq)


start = time.time()
svm = SVM()
svm.fit(data_train_val, label_train_val)
pred_train = svm.predict(data_train_val)
print("accu on train dataset: {}".format(accuracy_score(label_train_val, pred_train)))

pred = svm.predict(data_test_val)
print("accu on test dataset: {}".format(accuracy_score(label_test_val, pred)))
end = time.time()
print("time cost: {} seconds".format(end - start))

plt.plot(svm.losses, marker='o', linestyle='-')
plt.show()
print(svm.losses)
