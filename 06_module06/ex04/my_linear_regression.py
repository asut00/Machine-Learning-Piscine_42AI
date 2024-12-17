# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_linear_regression.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/18 16:52:26 by asuteau           #+#    #+#              #
#    Updated: 2024/06/18 16:52:27 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np


class MyLinearRegression():

	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas

	def loss_(self, y, y_hat):
		print 
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
	

if __name__=="__main__":
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	MyLR = MyLinearRegression
	lr1 = MyLR(np.array([[2], [0.7]]))
	
	# Example 0.0:
	y_hat = lr1.predict_(x)
	print(y_hat)
	# Output: array([[10.74695094], [17.05055804], [24.08691674], [36.24020866],[42.25621131]])

	# Example 0.1:
	print(lr1.loss_elem_(y, y_hat))
	# Output: array([[710.45867381], [364.68645485], [469.96221651], [108.97553412], [299.37111101]])


	# Example 0.2:
	print(lr1.loss_(y, y_hat))
	# Output: 195.34539903032385


	# Example 1.0:
	lr2 = MyLR(np.array([[1], [1]]), 5e-8, 1500000)
	lr2.fit_(x, y)
	print(f"lr2.thetas is {lr2.thetas}")
	# Output:
	# array([[1.40709365], [1.1150909 ]])


	# Example 1.1:
	y_hat = lr2.predict_(x)
	print(f"y_hat is {y_hat}")
	# Output:
	# array([[15.3408728 ],[25.38243697], [36.59126492], [55.95130097], [65.53471499]])


	# Example 1.2:
	print(lr2.loss_elem_(y, y_hat))
	# Output: array([[486.66604863], [115.88278416], [ 84.16711596], [ 85.96919719], [ 35.71448348]])


	# Example 1.3:
	print(lr2.loss_(y, y_hat))
	# Output: 80.83996294128525