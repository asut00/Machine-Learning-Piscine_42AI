# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    reg_linear_grad.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/21 12:21:56 by asuteau           #+#    #+#              #
#    Updated: 2024/06/21 12:21:57 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def add_intercept_(x):
	intercept = np.ones((x.shape[0], 1))
	X = np.hstack((intercept, x.reshape(-1, 1) if x.ndim == 1 else x))
	return X

def vec_reg_linear_grad(y, x, theta, lambda_):
	m = y.shape[0]
	Xprime = add_intercept_(x)
	theta_prime = theta.copy()
	theta_prime[0, 0] = 0.0
	y_hat = np.dot(Xprime, theta)
	gradient = ((np.dot(Xprime.T, (y_hat - y))) + (np.dot(lambda_, theta_prime))) / m
	return gradient



# truc chelous arguments

if __name__=="__main__":
	x = np.array([	[ -6, -7, -9],	[ 13, -2, 14],	[ -7, 14, -1],	[ -8, -4,	6],	[ -5, -9,	6],	[ 1, -5, 11],	[ 9, -11,	8]])
	y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	theta = np.array([[7.01], [3], [10.5], [-6]])
	
	# Example 1.1:
	# reg_linear_grad(y, x, theta, 1)

	# Example 1.2:
	print(vec_reg_linear_grad(y, x, theta, 1))

	# # Example 2.1:
	# # reg_linear_grad(y, x, theta, 0.5)

	# # Example 2.2:
	print(vec_reg_linear_grad(y, x, theta, 0.5))

	# # Example 3.1:
	# # reg_linear_grad(y, x, theta, 0.0)

	# # Example 3.2:
	print(vec_reg_linear_grad(y, x, theta, 0.0))
