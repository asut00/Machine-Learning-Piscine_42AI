# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_model_extended.py                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/21 10:35:07 by asuteau           #+#    #+#              #
#    Updated: 2024/06/21 10:35:08 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np


def add_polynomial_features(x, power):
	new_x = x
	for p in range(2, power + 1):
		for col in range(x.shape[1]): # pour chaque colonne
			new_x = np.hstack((new_x, (x[:,col] ** p).reshape(-1,1)))
	return (new_x)
		




if __name__=="__main__":

	x = np.arange(1,11).reshape(5, 2)

	# print(x)

	# # Example 1:
	print(add_polynomial_features(x, 3))
	# # Output:
	# # array([[1,2,1,4,[3,4,9,16,[5,6,25,36,[7,8,49,64,[9,10,81, 100,1,8],27,64],125, 216],343, 512],729, 1000]])

	# # Example 2:
	print(add_polynomial_features(x, 4))

	# # Output:
	# # array([[1,2,1,[3,4,9,[5,6,25,[7,8,49,[9,10,81,1,27,125,343,729,4,16,36,64,100,8,64,216,512,1000,1,16],81,256],625, 1296],2401, 4096]6561, 10000]]