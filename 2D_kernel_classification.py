'''
2D kernel classification using an RBF kernel and logistic regression:
generates synthetic blob data, maps it with an RBF kernel, learns weights via gradient descent,
and visualizes the nonlinear decision boundary over the 2D input space.
'''


import numpy as np
from math import e
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

# Generate the data
x_data, y_data = make_blobs(n_samples=100, centers=8, n_features=2,random_state=41)
new_y_data = [(1 if x % 2 != 0 else 0) for x in y_data]
y_data = np.array(new_y_data, dtype='float64') # (30,)
y_data = np.array([y_data])

si = y_data.shape[1] # Number of samples

stdev = 3
theta = np.full([si], 10.)
theta = theta.reshape((si, 1)) # (30,)

def find_h(y): # (1, 5)
    return 1 / (1 + np.exp(-y))

iterations = 300 # adjust this value to change how many iterations run
i = 0
alpha = 0.01
while i <= iterations:
    print(i)
    if(i == iterations):
        print("loading...")
    x_pre = []
    denom = 2*stdev*stdev
    counta = 0
    counta = 0
    while counta < x_data.shape[0]:
            countb = 0
            while countb < x_data.shape[0]:
                nume = (x_data[counta] - x_data[countb]).T @ (x_data[counta] - x_data[countb])
                # nume = - (abs(x_data[counta] - x_data[countb]))**2
                add = e**(-nume/denom)
                x_pre.append(add)
                countb += 1
            counta += 1
    # print(len(x_pre))
    # print(x_pre[0])
    x_mat = np.array(x_pre)
    # print(x_mat.shape)
    x_mat = x_mat.reshape((si,si)) # (30, 30)
    # print(x_mat.shape)
    y = x_mat @ theta # (30, 1)
    # print(y.shape)
    h = find_h(y) # (30, 1)
    # print(y.shape)

    # print(theta.shape) # (30, 1)
    gradient = h - y_data.T # (30, 1)
    # print(gradient.shape) # (30, 1)
    # print(x_mat.shape) # (30, 30)
    # print(gradient.shape) # (30, 1)
    gradient = x_mat @ gradient # (30, 1)
    # print(theta.shape) # (30, 1)
    theta -= alpha * gradient
    # print(theta.shape)
    i += 1

# Tester code

# x_test = np.linspace(-10,10,100)

x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1 # number
y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1 # number

x1_test, x2_test = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

x_tot_test = []

x_t = 0
y_t = 0
while x_t < x1_test.shape[0]:
    while y_t < x1_test.shape[1]:
        fill = np.array([x1_test[x_t][y_t], x2_test[x_t][y_t]])
        x_tot_test.append(fill)
        y_t += 1
    y_t = 0
    x_t += 1
x_tot_arr = np.array(x_tot_test)
# x_tot_arr = np.c_[x1_test.ravel(), x2_test.ravel()]
# print(x_tot_arr[100:105])

x_pre2 = []
denom = 2*stdev*stdev
counta = 0
countb = 0
while counta < x_tot_arr.shape[0]:
    countb = 0
    while countb < x_data.shape[0]:
        nume = (x_tot_arr[counta] - x_data[countb]).T @ (x_tot_arr[counta] - x_data[countb])
        # print(nume)
        add = e**(-nume/denom)
        x_pre2.append(add)
        countb += 1
    counta += 1
x_mat2 = np.array(x_pre2)
x_mat2 = x_mat2.reshape((x_tot_arr.shape[0],si))
# print(x_mat2[:5,:5])

y_test = x_mat2 @ theta
h_test = find_h(y_test) # (33196, 1)
h_test = h_test.reshape(x1_test.shape[0], x1_test.shape[1])
plt.contourf(x1_test, x2_test, h_test, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)

plt.tight_layout()
plt.show()
