# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_pred.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/20 10:03:35 by asuteau           #+#    #+#              #
#    Updated: 2024/06/20 10:03:37 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def add_intercept(x):
	intercept = np.ones((x.shape[0], 1))
	X = np.hstack((intercept, x.reshape(-1, 1) if x.ndim == 1 else x))
	return X

def sigmoid_(x):
	res = 1 / (1 + np.exp(-x))
	return res

def logistic_predict_(x, theta):

	Xprime = add_intercept(x)
	prediction = np.dot(Xprime, theta)
	res = sigmoid_(prediction)
	return res


if __name__=="__main__":
	# Example 1
	x = np.array([4]).reshape((-1, 1))
	theta = np.array([[2], [0.5]])
	print(logistic_predict_(x, theta))
	# Output: array([[0.98201379]])

	# Example 1
	x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
	theta2 = np.array([[2], [0.5]])
	print(logistic_predict_(x2, theta2))
	# Output:	array([[0.98201379],[0.99624161],[0.97340301],[0.99875204],[0.90720705]])

	# Example 3
	x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
	theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	print(logistic_predict_(x3, theta3))
	# Output:	array([[0.03916572],	[0.00045262],	[0.2890505 ]])