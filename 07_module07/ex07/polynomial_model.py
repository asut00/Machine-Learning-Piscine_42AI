# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_model.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/19 14:11:40 by asuteau           #+#    #+#              #
#    Updated: 2024/06/19 14:11:41 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def add_polynomial_features(x, power):
	res = np.zeros((len(x), power))
	for i in range(len(x)): # pour chaque ligne de x
		for j in range(1, power + 1): # pour chaque power
			# print(f"i is {i}")
			res[i][j - 1] = x[i] ** j
	return res


if __name__=="__main__":

	x = np.arange(1,6).reshape(-1, 1)

	print(x)

	# Example 0:
	re = add_polynomial_features(x, 3)
	print(re)
	# Output: array([[ 1, 1, 1],[ 2, 4, 8],[ 3, 9, 27],[ 4, 16, 64],[ 5, 25, 125]])

	# Example 1:
	re = add_polynomial_features(x, 6)
	print(re)
	# Output:
	# array([[ 1, 1, 1, 1, 1, 1],
	# [ 2, 4, 8, 16, 32, 64],
	# [ 3, 9, 27, 81, 243, 729],
	# [ 4, 16, 64, 256, 1024, 4096],
	# [ 5, 25, 125, 625, 3125, 15625]])