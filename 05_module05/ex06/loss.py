# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    loss.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/17 16:17:16 by asuteau           #+#    #+#              #
#    Updated: 2024/06/17 16:17:17 by asuteau          ###   ########.fr        #
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

def predict_(x, theta):
	X = add_intercept(x)
	res = np.zeros(x.shape[0]) # le resultat est un array unidemensionel du mm nbre de ligne que x
	for i in range(X.shape[0]): # pour chaque ligne 
		for j in range(X.shape[1]): # pour chaque colonne
			thetaj = theta[j] 
			xj = x[j]
			res[i] += theta[j] * X[i][j]# pour la ligne actuelle de X, on multiplie la valeur qui se trouve a la colonne j dans x par la valeur qui se trouve a la ligne j dans theta
	return (res)


def loss_elem_(y, y_hat):
	res = np.zeros(y_hat.shape[0])
	for i in range(y_hat.shape[0]):
		res[i] = (y_hat[i] - y[i]) ** 2
	return res

def loss_(y, y_hat):
	le = loss_elem_(y, y_hat)
	res = 0
	for elem in le:
		res += elem
	return (res / (2 * len(le)))




if __name__=="__main__":
	x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
	theta1 = np.array([[2.], [4.]])
	y_hat1 = predict_(x1, theta1)
	# (())
	y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

	# Example 1:
	print(loss_elem_(y1, y_hat1))
	# Output: array([[0.], [1], [4], [9], [16]])

	# Example 2:
	print(loss_(y1, y_hat1))
	# Output: 3.0


	x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
	theta2 = np.array(np.array([[0.], [1.]]))
	y_hat2 = predict_(x2, theta2)
	y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

	# Example 3:
	print(loss_(y2, y_hat2))
	# Output: 2.142857142857143

	# Example 4:
	print(loss_(y2, y2))
	# Output: 0.0