# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    data_splitter.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/19 17:42:24 by asuteau           #+#    #+#              #
#    Updated: 2024/06/19 17:42:26 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np


def data_splitter(x, y, proportion):
	# rand = np.random.shuffle(x)
	# print(rand)
	permutation = np.random.permutation(len(x))

	x_shf = x[permutation]
	y_shf = y[permutation]

	len_train = int(len(x) * proportion)
	len_test = int(len(x) - len_train)

	# x_train = np.zeros((int(len(x)*proportion), int(len(x[0]))))
	# x_test = np.zeros((int(len(x) - (len(x)*proportion)), int(len(x[0]))))
	# y_train = np.zeros((int(len(x)*proportion), int(len(x[0]))))
	# y_test = np.zeros((int(len(x) - (len(x)*proportion)), int(len(x[0]))))

	# Séparer les données d'entraînement et de test
	x_train = x_shf[:len_train]
	y_train = y_shf[:len_train]
	x_test = x_shf[len_train:]
	y_test = y_shf[len_train:]

	# for i in range(len(x)):
	# 	if (i < len_train):
	# 		x_train[i] = x[i]
	# 		y_train[i] = y[i]
	# 	if (i >= len_train):
	# 		x_test[i - len_train] = x[i]
	# 		y_test[i - len_train] = y[i]

	return x_train, x_test, y_train, y_test


if __name__=="__main__":
	x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
	y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

	# # Example 1:
	res01 = data_splitter(x1, y, 0.8)
	print(res01)
	# # Output: (array([ 1, 59, 42, 300]), array([10]), array([0, 0, 1, 0]), array([1]))

	# # Example 2:
	print(data_splitter(x1, y, 0.5))
	# # Output:
	# # (array([59, 10]), array([ 1, 300, 42]), array([0, 1]), array([0, 0, 1]))

	x2 = np.array([[ 1, 42],
	[300, 10],
	[ 59, 1],
	[300, 59],
	[ 10, 42]])

	y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

	# Example 3:
	print(data_splitter(x2, y, 0.8))
	# Output:
	# (array([[ 10, 42],
	# [300, 59],
	# [ 59, 1],
	# [300, 10]]),
	# array([[ 1, 42]]),
	# array([0, 1, 0, 1]),
	# array([0]))


	# Example 4:
	print(data_splitter(x2, y, 0.5))
	# Output:
	# (array([[59, 1],[10, 42]]),	array([[300, 10],	[300, 59],	[ 1, 42]]),	array([0, 0]),	array([1, 1, 0]))