# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    log_loss.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/20 10:14:44 by asuteau           #+#    #+#              #
#    Updated: 2024/06/20 10:14:45 by asuteau          ###   ########.fr        #
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

def log_loss_(y, y_hat, eps=1e-15):
	elems = y * np.log(y_hat) + ((1 - y) * np.log(1 - y_hat))
	sum = np.sum(elems)
	res = -(1/len(y)) * sum
	return res

if __name__=="__main__":

	# Example 1:
	y1 = np.array([1]).reshape((-1, 1))
	x1 = np.array([4]).reshape((-1, 1))
	theta1 = np.array([[2], [0.5]])
	y_hat1 = logistic_predict_(x1, theta1)
	print(log_loss_(y1, y_hat1))
	# Output:	0.01814992791780973

	# Example 2:
	y2 = np.array([[1], [0], [1], [0], [1]])
	x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
	theta2 = np.array([[2], [0.5]])
	y_hat2 = logistic_predict_(x2, theta2)
	print(log_loss_(y2, y_hat2))
	# Output:	2.4825011602474483

	# Example 3:
	y3 = np.array([[0], [1], [1]])
	x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
	theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	y_hat3 = logistic_predict_(x3, theta3)
	print(log_loss_(y3, y_hat3))
	# Output:	2.9938533108607053