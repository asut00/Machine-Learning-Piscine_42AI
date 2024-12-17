# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    fit.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/19 11:53:24 by asuteau           #+#    #+#              #
#    Updated: 2024/06/19 11:53:27 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def add_intercept(x):
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


def transpose(x):
	t_x = np.zeros((len(x[0]),len(x)))
	for i in range(len(x)): # pour chaque ligne
		for j in range(len(x[i])): # pour chaque colonne
			t_x[j][i] = x[i][j]
	return (t_x)


def gradient(x, y, theta):
	Xprime = add_intercept(x)
	XprimeT = transpose(Xprime)
	res = (1 / len(x)) * np.dot(XprimeT, (np.dot(Xprime, theta) - y))
	return res


def fit_(x, y, theta, alpha, max_iter):
	for i in range(max_iter):
		theta = theta - alpha * gradient(x, y, theta)
	return theta


if __name__=="__main__":
	x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
	y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
	theta = np.array([[42.], [1.], [1.], [1.]])
	# Example 0:
	theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
	print(theta2)
	# Output: array([[41.99..],[0.97..], [0.77..], [-1.20..]])
	# Example 1:
	# predict_(x, theta2)
	# Output: array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])