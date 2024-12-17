# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/19 10:30:21 by asuteau           #+#    #+#              #
#    Updated: 2024/06/19 10:30:22 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def simple_predict_verbose(x, theta):
	prediction = np.zeros(len(x))
	for i in range(len(x)): # pour chaque ligne de x avec toutes ses features
		prediction[i] = theta[0] # on initialise la prediction avec la constante theta[0]
		for j in range(0, len(x[0])): # pour chaque colonne de x (il y en a 3)
			prediction[i] += theta[j + 1] * x[i][j] # on multiplie la feature / valeur de x correspondante et on l'ajoute a la valeur de prediction pour ce vecteur (ensemble de valeurs de features) de x
	return prediction


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


def simple_predict(x, theta):
	Xprime = add_intercept(x)
	# print(f"Xprime is :\n{Xprime}")
	# print()
	prediction = np.dot(Xprime, theta)
	return prediction


if __name__=="__main__":
	x = np.arange(1,13).reshape((4,-1))

	# Example 1:
	theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))

	# print(x)
	# print(theta1)
	# print()

	print(simple_predict(x, theta1))
	# Ouput: array([[5.], [5.], [5.], [5.]])
	# Do you understand why y_hat contains only 5’s here?

	# Example 2:
	theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
	print(simple_predict(x, theta2))
	# Output: array([[ 1.], [ 4.], [ 7.], [10.]])
	# Do you understand why y_hat == x[:,0] here?

	# Example 3:
	theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
	print(simple_predict(x, theta3))
	# Output: array([[ 9.64], [24.28], [38.92], [53.56]])

	# Example 4:
	theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
	print(simple_predict(x, theta4))
	# Output:	array([[12.5], [32. ], [51.5], [71. ]])