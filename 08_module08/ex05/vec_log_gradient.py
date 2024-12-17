# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_log_gradient.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/20 11:16:43 by asuteau           #+#    #+#              #
#    Updated: 2024/06/20 11:16:45 by asuteau          ###   ########.fr        #
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


def vec_log_gradient(x, y, theta):
	Xprime = add_intercept(x)
	XprimeT = Xprime.T

	l_pred = logistic_predict_(x, theta)
	# print(f"{XprimeT * (sigmoid_(x) - y)}")
	res = (1/len(x)) * np.dot(XprimeT, (l_pred - y))
	return (res)



if __name__=="__main__":

	# Example 1:
	y1 = np.array([1]).reshape((-1, 1))
	x1 = np.array([4]).reshape((-1, 1))
	theta1 = np.array([[2], [0.5]])
	print(log_gradient(x1, y1, theta1))
	# Output:	array([[-0.01798621],	[-0.07194484]])

	# Example 2:
	y2 = np.array([[1], [0], [1], [0], [1]])
	x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
	theta2 = np.array([[2], [0.5]])
	print(log_gradient(x2, y2, theta2))
	# Output:array([[0.3715235 ],[3.25647547]])

	# Example 3:
	y3 = np.array([[0], [1], [1]])
	x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
	theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	print(log_gradient(x3, y3, theta3))
	# Output:	array([[-0.55711039],[-0.90334809],[-2.01756886],[-2.10071291],[-3.27257351]])