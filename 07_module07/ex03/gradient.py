# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    gradient.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/19 11:41:00 by asuteau           #+#    #+#              #
#    Updated: 2024/06/19 11:41:01 by asuteau          ###   ########.fr        #
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


if __name__=="__main__":
	x = np.array([
	[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]])
	y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	theta1 = np.array([0, 3, 0.5, -6]).reshape((-1, 1))


	# Example :
	print(gradient(x, y, theta1))
	# Output: array([[ -33.71428571], [ -37.35714286], [183.14285714], [-393.]])


	# Example :
	theta2 = np.array([0, 0, 0, 0]).reshape((-1, 1))
	print(gradient(x, y, theta2))
	# Output: array([[ -0.71428571], [ 0.85714286], [23.28571429], [-26.42857143]])