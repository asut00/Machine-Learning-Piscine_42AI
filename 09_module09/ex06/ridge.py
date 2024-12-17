# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ridge.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/21 13:19:36 by asuteau           #+#    #+#              #
#    Updated: 2024/06/21 13:19:37 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class MyRidge():

	def __init__(self, thetas, alpha=0.001, max_iter=100000, lambda_=0.5):
		self.thetas = thetas
		self.alpha = alpha
		self.max_iter = max_iter
		self.lambda_ = lambda_

	def add_intercept(self, x):
		intercept = np.ones((x.shape[0], 1))
		X = np.hstack((intercept, x.reshape(-1, 1) if x.ndim == 1 else x))
		return X

	def l2(self, theta):
		theta_copy = self.thetas.copy()
		theta_copy[0, 0] = 0
		regularization = np.dot(theta_copy.T, theta_copy)
		return float(regularization[0, 0])

	def loss_elem_(self, y, y_hat):
		loss_elem = np.dot((y_hat - y).T,(y_hat - y))
		return loss_elem

	def loss_(self, y, y_hat):
		const = (1 / (2 * len(y)))
		loss_elems_ = self.loss_elem_(y, y_hat)
		res = const * (loss_elems_ + self.lambda_ * (self.l2(self.thetas)))
		return res

	def predict_(self, x):
		prediction = np.dot(self.add_intercept(x), self.thetas)
		return prediction

	def gradient_(self, x, y):
		m = y.shape[0]
		Xprime = self.add_intercept(x)
		theta_prime = self.thetas.copy()
		theta_prime[0, 0] = 0.0
		y_hat = np.dot(Xprime, self.thetas)
		gradient = ((np.dot(Xprime.T, (y_hat - y))) + (np.dot(self.lambda_, theta_prime))) / m
		return gradient
	
	def fit_(self, x, y):
		for i in range(self.max_iter):
			self.thetas = self.thetas - self.alpha * self.gradient_(x, y)
		return self.thetas
	


if __name__=="__main__":

	x = np.array([	[ -6, -7, -9],	[ 13, -2, 14],	[ -7, 14, -1],	[ -8, -4,	6],	[ -5, -9,	6],	[ 1, -5, 11],	[ 9, -11,	8]])
	y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	
	theta = np.array([[7.01], [3], [10.5], [-6]])
	
	myridge = MyRidge(theta, lambda_=1)

	print(myridge.gradient_(x, y))

	myridge.fit_(x, y)

	# Example 1.1:
	# reg_linear_grad(y, x, theta, 1)

	# Example 1.2:
	# print(vec_reg_linear_grad(y, x, theta, 1))

	# # Example 2.1:
	# # reg_linear_grad(y, x, theta, 0.5)

	# # Example 2.2:

	# print(vec_reg_linear_grad(y, x, theta, 0.5))

	# # Example 3.1:
	# # reg_linear_grad(y, x, theta, 0.0)

	# # Example 3.2:

	# print(vec_reg_linear_grad(y, x, theta, 0.0))
