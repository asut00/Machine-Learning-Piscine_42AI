# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    fit.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/18 16:06:00 by asuteau           #+#    #+#              #
#    Updated: 2024/06/18 16:06:01 by asuteau          ###   ########.fr        #
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



def vec_gradient(x, y, theta):
	Xprime = add_intercept(x)
	XprimeT = transpose(Xprime)
	nablaj = 1/len(x) * np.dot(XprimeT, np.dot(Xprime, theta) - y)
	return nablaj


def fit_(x, y, theta, alpha, max_iter):
	new_theta = np.copy(theta)
	for i in range(max_iter):
		gradient = vec_gradient(x, y, new_theta)
		new_theta = new_theta - alpha * gradient
	return new_theta


if __name__=="__main__":
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	theta= np.array([1, 1]).reshape((-1, 1))

	# Example 0:
	theta1 = fit_(x, y, theta, alpha=5e-8,  max_iter=1500000)
	print(theta1)
	# Output: array([[1.40709365],[1.1150909 ]])

	# Example 1:
	# predict(x, theta1)
	# Output: array([[15.3408728 ], 	[25.38243697],	[36.59126492],	[55.95130097],	[65.53471499]])