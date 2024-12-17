# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    loss.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/19 11:12:17 by asuteau           #+#    #+#              #
#    Updated: 2024/06/19 11:12:19 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np


def transpose(x):
	t_x = np.zeros((len(x[0]),len(x)))
	for i in range(len(x)): # pour chaque ligne
		for j in range(len(x[i])): # pour chaque colonne
			t_x[j][i] = x[i][j]
	return (t_x)

def loss_(y, y_hat):
	res = 1 / (2 * len(y)) * np.dot(transpose(y_hat - y), (y_hat - y))
	return res

if __name__=="__main__":
	import numpy as np
	X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
	Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	print(f"y is {X}")
	print(f"y_hat is {Y}")
	# Example 1:
	print(loss_(X, Y))
	# Output: 2.142857142857143
	# Example 2:
	print(loss_(X, X))
	# Output: 0.0
