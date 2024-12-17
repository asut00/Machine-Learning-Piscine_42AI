# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/17 15:20:26 by asuteau           #+#    #+#              #
#    Updated: 2024/06/17 15:20:27 by asuteau          ###   ########.fr        #
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


if __name__=="__main__":
	x = np.arange(1,6)


	# Example 1:
	theta1 = np.array([[5], [0]])
	print(predict_(x, theta1))
	# # Ouput:
	# array([[5.], [5.], [5.], [5.], [5.]])
	# # Do you remember why y_hat contains only 5’s here?

	# Example 2:
	theta2 = np.array([[0], [1]])
	print(predict_(x, theta2))
	# Output:
	# array([[1.], [2.], [3.], [4.], [5.]])
	# Do you remember why y_hat == x here?

	# Example 3:
	theta3 = np.array([[5], [3]])
	print(predict_(x, theta3))
	# Output:
	# array([[ 8.], [11.], [14.], [17.], [20.]])

	# Example 4:
	theta4 = np.array([[-3], [1]])
	print(predict_(x, theta4))
	# Output:
	# array([[-2.], [-1.], [ 0.], [ 1.], [ 2.]]