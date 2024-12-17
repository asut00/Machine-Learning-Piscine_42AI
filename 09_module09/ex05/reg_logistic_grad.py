# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    reg_logistic_grad.py                               :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/21 13:03:56 by asuteau           #+#    #+#              #
#    Updated: 2024/06/21 13:03:58 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np


def add_intercept_(x):
	intercept = np.ones((x.shape[0], 1))
	X = np.hstack((intercept, x.reshape(-1, 1) if x.ndim == 1 else x))
	return X


def sigmoid_(x):
	res = 1 / (1 + np.exp(-x))
	return res


def vec_reg_logistic_grad(y, x, theta, lambda_):
	m = y.shape[0]
	Xprime = add_intercept_(x)
	theta_prime = theta.copy()
	theta_prime[0, 0] = 0.0
	y_hat = sigmoid_(np.dot(Xprime, theta))
	gradient = ((np.dot(Xprime.T, (y_hat - y))) + (np.dot(lambda_, theta_prime))) / m
	return gradient


if __name__=="__main__":
	x = np.array([[0, 2, 3, 4],
	[2, 4, 5, 5],
	[1, 3, 2, 7]])
	y = np.array([[0], [1], [1]])
	theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	# Example 1.1:
	# reg_logistic_grad(y, x, theta, 1)

	# Example 1.2:
	print(vec_reg_logistic_grad(y, x, theta, 1))

	# Example 2.1:
	# reg_logistic_grad(y, x, theta, 0.5)

	# Example 2.2:
	print(vec_reg_logistic_grad(y, x, theta, 0.5))

	# Example 3.1:
	# reg_logistic_grad(y, x, theta, 0.0)

	# Example 3.2:
	print(vec_reg_logistic_grad(y, x, theta, 0.0))
