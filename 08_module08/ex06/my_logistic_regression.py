# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_logistic_regression                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/20 11:21:30 by asuteau           #+#    #+#              #
#    Updated: 2024/06/20 11:21:31 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np


class MyLogisticRegression():
	
	def __init__(self, theta, alpha=0.001, max_iter=1000):
		self.theta = theta
		self.alpha = alpha
		self.max_iter = max_iter
		
	def sigmoid_(self, x):
		res = 1 / (1 + np.exp(-x))
		return res

	def add_intercept_(self, x):
		intercept = np.ones((x.shape[0], 1))
		X = np.hstack((intercept, x.reshape(-1, 1) if x.ndim == 1 else x))
		return X

	def predict_(self, x):
		Xprime = self.add_intercept_(x)
		prediction = np.dot(Xprime, self.theta)
		res = self.sigmoid_(prediction)
		return res

	def gradient_(self, x, y, theta):
		Xprime = self.add_intercept_(x)
		XprimeT = Xprime.T
		l_pred = self.predict_(x)
		res = (1/len(x)) * np.dot(XprimeT, (l_pred - y))
		return (res)

	def fit_(self, x, y):
		for i in range(self.max_iter):
			self.theta = self.theta - self.alpha * self.gradient_(x, y, self.theta)

	def loss_(self, y, y_hat, eps=1e-15):
		elems = y * np.log(y_hat + eps) + ((1 - y) * np.log(1 - y_hat + eps))
		sum = np.sum(elems)
		res = -(1/len(y)) * sum
		return res

if __name__=="__main__":

	MyLR = MyLogisticRegression

	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
	Y = np.array([[1], [0], [1]])

	thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
	mylr = MyLR(thetas)

	# Example 0:
	p = mylr.predict_(X)
	print(p)
	# Output:	array([[0.99930437],	[1. ],	[1. ]])

	# Example 1:
	q = mylr.loss_(Y,p)
	print(q)
	# Output: 11.513157421577002

	# Example 2:
	mylr.fit_(X, Y)
	print(mylr.theta)
	# Output:	array([[ 2.11826435][ 0.10154334][ 6.43942899][-5.10817488][ 0.6212541 ]])

	# Example 3:
	pred = mylr.predict_(X)
	print(pred)
	# Output:	array([[0.57606717]	[0.68599807]	[0.06562156]])
	
	# Example 4:
	l = mylr.loss_(Y,pred)
	print(l)
	# Output:	1.4779126923052268