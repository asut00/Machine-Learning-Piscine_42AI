# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_gradient.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/18 11:28:31 by asuteau           #+#    #+#              #
#    Updated: 2024/06/18 11:28:33 by asuteau          ###   ########.fr        #
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


# def simple_gradient(x, y, theta):
# 	loss_elems = theta[0] + theta[1] * x
# 	print (loss_elems)	
# 	nabla0_diff_sum = np.sum(loss_elems - y)
# 	nabla0 = nabla0_diff_sum / len(x)
# 	print(nabla0)
# 	nabla1_diff_sum = np.sum((loss_elems - y) * x)
# 	nabla1 = nabla1_diff_sum / len(x)
# 	print(nabla1)



def vec_gradient(x, y, theta):
	Xprime = add_intercept(x)
	XprimeT = transpose(Xprime)
	# print(f"x is {x}")
	# print()
	# print(f"theta is {theta}")
	# print()
	
	# print(f"XprimeT is {XprimeT}")
	# print()
	
	# Compute the gradient
	nablaj = 1/len(x) * np.dot(XprimeT, np.dot(Xprime, theta) - y)
	return nablaj




if __name__=="__main__":
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))


	# Example 0:
	theta1 = np.array([2, 0.7]).reshape((-1, 1))
	# print(theta1)
	# transpose(1)

	print(vec_gradient(x, y, theta1))
	# # Output: array([[-19.0342...], [-586.6687...]])

	# # Example 1:
	# theta2 = np.array([1, -0.4]).reshape((-1, 1))
	# gradient(x, y, theta2)
	# # Output: array([[-57.8682...], [-2230.1229...]])