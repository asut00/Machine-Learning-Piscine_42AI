# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    tools.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/17 14:28:21 by asuteau           #+#    #+#              #
#    Updated: 2024/06/17 14:28:22 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def add_intercept(x):
	if not isinstance(x, np.ndarray):
		raise TypeError ("Error: arg must be Numpy array")
	if len(x.shape) > 2 and x.shape[2] != 1:
		raise TypeError ("Vector must be at most 2 dimensional")
	# print(x.shape)
	if len(x.shape) == 1:
		xshape1 = 1
	else:
		xshape1 = x.shape[1]
	X = np.ones((x.shape[0], xshape1 + 1))
	# print(f"X is {X}")
	# print(f"x is {x}")
	# print(f"x.shape[0] is {x.shape[0]}")
	# print(f"xshape1 is {xshape1}")
	# print(f"xshape1 is {xshape1}")
	for j in range(1, xshape1 + 1): # pour chaque colonne a partie de la deuxieme
		# print("here")
		for i in range(0, x.shape[0]): # pour chaque ligne 
			# print(f"i is {i} and j is {j}")
			# print(f"X[i][j] is {X[i][j]}\n")
			if xshape1 == 1: # si le tableau x est one dimensional
				X[i][j] = x[i]
			else: # si le tableau x est bi dimensional
				X[i][j] = x[i][j-1]
			# print(f"after operation X[i][j] is {X[i][j]}")
	# print(X)
	return X
			
	


if __name__=="__main__":
	x = np.arange(1, 6)
	print(add_intercept(x))
	
# Example 2:
	y = np.arange(1,10).reshape((3,3))

	print(add_intercept(y))

	# Output:
	# array([[1., 1., 2., 3.],
	# [1., 4., 5., 6.],
	# [1., 7., 8., 9.]])