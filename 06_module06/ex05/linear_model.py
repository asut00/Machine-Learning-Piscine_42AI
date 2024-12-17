# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_model.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/18 17:43:38 by asuteau           #+#    #+#              #
#    Updated: 2024/06/18 17:43:39 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# from my_linear_regression import MyLinearRegression as MyLR


import matplotlib.pyplot as plt
import numpy as np


class MyLinearRegression():

	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas

	def mse_(self, y, y_hat):
		loss_elems = (y_hat - y) ** 2
		mse = np.sum(loss_elems) / len(y_hat)
		return mse

	def loss_(self, y, y_hat):
		le = np.dot(self.transpose(y_hat - y), y_hat - y)
		print(f"le is {le}")
		le_sum = np.sum(le)
		print(f"len(y_hat is {len(y_hat)})")
		print(f"le_sum is {le_sum}")
		return (le_sum / (2 * len(y_hat)))

	def loss_elem_(self, y, y_hat):
		res = np.zeros(y_hat.shape[0])
		for i in range(y_hat.shape[0]):
			res[i] = (y_hat[i] - y[i]) ** 2
		return res

	def add_intercept(self, x):
		if not isinstance(x, np.ndarray):
			raise TypeError ("Error: arg must be Numpy array")
		if len(x.shape) > 2 and x.shape[2] != 1:
			raise TypeError ("Vector must be at most 2 dimensional")
		if len(x.shape) == 1:
			xshape1 = 1
		else:
			xshape1 = x.shape[1]
		X = np.ones((x.shape[0], xshape1 + 1))
		for j in range(1, xshape1 + 1): # pour chaque colonne a partie de la deuxieme
			for i in range(0, x.shape[0]): # pour chaque ligne 
				if xshape1 == 1: # si le tableau x est one dimensional
					X[i][j] = x[i]
				else: # si le tableau x est bi dimensional
					X[i][j] = x[i][j-1]
		return X

	def transpose(self, x):
		t_x = np.zeros((len(x[0]),len(x)))
		for i in range(len(x)): # pour chaque ligne
			for j in range(len(x[i])): # pour chaque colonne
				t_x[j][i] = x[i][j]
		return (t_x)

	def vec_gradient(self, x, y, theta):
		Xprime = self.add_intercept(x)
		XprimeT = self.transpose(Xprime)
		nablaj = 1/len(x) * np.dot(XprimeT, np.dot(Xprime, theta) - y)
		return nablaj


	def fit_(self, x, y):
		new_theta = np.copy(self.thetas)
		for i in range(self.max_iter):
			gradient = self.vec_gradient(x, y, new_theta)
			new_theta = new_theta - self.alpha * gradient
		self.thetas = new_theta
		return self.thetas

	def predict_(self, x):
		# print(self.thetas)
		# print(self.add_intercept(x))
		y_hat = np.dot(self.add_intercept(x), self.thetas)
		return y_hat


def plot(x, y, y_pred, theta):
	# y_pred = self.thetas + self.thetas * x
	plt.scatter(x, y, color='blue', label='Data points')
	plt.plot(x, y_pred, color='red', label='Prediction line')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.show()
	


if __name__=="__main__":

	MyLR = MyLinearRegression

	data = pd.read_csv("./are_blue_pills_magics.csv")
	Xpill = np.array(data['Micrograms']).reshape(-1,1)
	
	Yscore = np.array(data['Score']).reshape(-1,1)

	linear_model1 = MyLR(np.array([[89.0], [-8]]))
	linear_model2 = MyLR(np.array([[89.0], [-6]]))

	Y_model1 = linear_model1.predict_(Xpill)
	Y_model2 = linear_model2.predict_(Xpill)



	print(MyLR.mse_(Yscore, Y_model1))	# 57.60304285714282
	print(mean_squared_error(Yscore, Y_model1))  # 57.603042857142825
	print(MyLR.mse_(Yscore, Y_model2)) # 232.16344285714285
	print(mean_squared_error(Yscore, Y_model2)) # 232.16344285714285

	plot(Xpill, Yscore, Y_model1, linear_model1.thetas)