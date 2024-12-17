# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    other_losses.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/17 17:08:02 by asuteau           #+#    #+#              #
#    Updated: 2024/06/17 17:08:03 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np

def mse_(y, y_hat):
	loss_elems = (y_hat - y) ** 2
	mse = np.sum(loss_elems) / len(y_hat)
	return mse

def rmse_(y, y_hat):
	return np.sqrt(mse_(y, y_hat))

def mae_(y, y_hat):
	loss_elems_abs = abs(y_hat - y)
	mae = np.sum(loss_elems_abs) / len(y_hat)
	return mae

def r2score_(y,y_hat):
	mean_y = np.sum(y) / len(y)
	div = np.sum((y_hat - y) ** 2) / np.sum((y - mean_y) ** 2)
	r2score = 1 - div
	return r2score




if __name__=="__main__":
	# Example 1:
	x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
	y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

	# Mean squared error
	## your implementation
	print(mse_(x,y))
	## Output: 4.285714285714286

	# Root mean squared error
	## your implementation
	print(rmse_(x,y))

	print(mae_(x,y))

	print(r2score_(x,y))