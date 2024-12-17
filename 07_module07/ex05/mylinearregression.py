# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mylinearregression.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/19 12:02:25 by asuteau           #+#    #+#              #
#    Updated: 2024/06/19 12:02:26 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np


class MyLinearRegression:

	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.thetas = thetas
		self.alpha = alpha
		self.max_iter = max_iter

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


	def loss_(self, y, y_hat):
		res = 1 / (2 * len(y)) * np.dot(self.transpose(y_hat - y), (y_hat - y))
		return res

	# ancien def loss :
	# def loss_(self, y, y_hat):
	# 	le = np.dot(self.transpose(y_hat - y), y_hat - y)
	# 	le_sum = np.sum(le)
	# 	return (le_sum / (2 * len(y_hat)))


	def loss_elem_(self, y, y_hat):
		res = np.zeros(y_hat.shape[0])
		for i in range(y_hat.shape[0]):
			res[i] = (y_hat[i] - y[i]) ** 2
		return res


	def gradient(self, x, y, theta):
		Xprime = self.add_intercept(x)
		XprimeT = self.transpose(Xprime)
		res = (1 / len(x)) * np.dot(XprimeT, (np.dot(Xprime, theta) - y))
		return res


	def fit_(self, x, y):
		for i in range(self.max_iter):
			self.thetas = self.thetas - self.alpha * self.gradient(x, y, self.thetas)
		return self.thetas
	
	def predict_(self, x):
		prediction = np.dot(self.add_intercept(x), self.thetas)
		return (prediction)


if __name__=="__main__":
	MyLR = MyLinearRegression
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
	Y = np.array([[23.], [48.], [218.]])
	mylr = MyLR([[1.], [1.], [1.], [1.], [1]])


	# Example 0:
	y_hat = mylr.predict_(X)
	print(y_hat)
	print()
	# Output:	array([[8.], [48.], [323.]])


	# Example 1:
	print(f"mylr.loss_elem_(Y, y_hat) is {mylr.loss_elem_(Y, y_hat)}")
	# Output:array([[225.], [0.], [11025.]])


	# Example 2:
	print(f"mylr.loss_(Y, y_hat) is {mylr.loss_(Y, y_hat)}")
	# Output:	1875.0


	# Example 3:
	mylr.alpha = 1.6e-4
	mylr.max_iter = 200000
	mylr.fit_(X, Y)
	print(mylr.thetas)
	print()
	# Output:	array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])

	# Example 4:
	y_hat = mylr.predict_(X)
	print(y_hat)
	# Output:	array([[23.417..], [47.489..], [218.065...]])


	# Example 5:
	print(f"mylr.loss_elem_(Y, y_hat) is {mylr.loss_elem_(Y, y_hat)}")
	# Output:	array([[0.174..], [0.260..], [0.004..]])


	# Example 6:
	print(f"mylr.loss_(Y, y_hat) is {mylr.loss_(Y, y_hat)}")
	# Output: 0.0732..