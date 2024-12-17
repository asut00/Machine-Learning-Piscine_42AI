# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linear_loss_reg.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/21 11:51:37 by asuteau           #+#    #+#              #
#    Updated: 2024/06/21 11:51:38 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def l2(theta):
	theta_copy = theta.copy()
	theta_copy[0, 0] = 0
	regularization = np.dot(theta_copy.T, theta_copy)
	return float(regularization[0, 0])

def reg_loss_(y, y_hat, theta, lambda_):
	res = (1 / (2 * len(y))) * (np.dot((y_hat - y).T,(y_hat - y)) + lambda_ * (l2(theta)))
	return res

if __name__=="__main__":

	y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
	theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
	# Example :
	print(reg_loss_(y, y_hat, theta, .5))
	# Output:	0.8503571428571429
	# Example :
	print(reg_loss_(y, y_hat, theta, .05))
	# Output:	0.5511071428571429
	# Example :
	print(reg_loss_(y, y_hat, theta, .9))
	# Output:	1.116357142857143