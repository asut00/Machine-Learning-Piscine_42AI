# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vec_loss.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/17 16:35:35 by asuteau           #+#    #+#              #
#    Updated: 2024/06/17 16:35:36 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np


# def plot(x, y, theta):
# 	y_pred = theta[0] + theta[1] * x
# 	plt.scatter(x, y, color='blue', label='Data points')
# 	plt.plot(x, y_pred, color='red', label='Prediction line')
# 	plt.xlabel('x')
# 	plt.ylabel('y')
# 	plt.legend()
# 	plt.show()

def loss_(y, y_hat):
	# le = (y_hat - y) ** 2
	le = np.dot(y_hat - y, y_hat - y)
	le_sum = np.sum(le)
	return (le_sum / (2 * len(y_hat)))



if __name__=="__main__":
	X = np.array([0, 15, -9, 7, 12, 3, -21])
	Y = np.array([2, 14, -13, 5, 12, 4, -19])
	
	# Example 1:
	print(loss_(X, Y))
	# Output:2.142857142857143
	
	# Example 2:
	print(loss_(X, X))
	# Output:0.0