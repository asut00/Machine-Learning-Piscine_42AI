# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logistic_loss_reg.py                               :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/21 12:09:47 by asuteau           #+#    #+#              #
#    Updated: 2024/06/21 12:09:48 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def l2(theta):
	theta_copy = theta.copy()
	theta_copy[0, 0] = 0
	regularization = np.dot(theta_copy.T, theta_copy)
	return float(regularization[0, 0])

def reg_log_loss_(y, y_hat, theta, lambda_):
	
	const1 = -(1 / len(y))
	ones = np.ones(y.shape)
	log_loss_elems = np.dot(y.T, np.log(y_hat)) + np.dot((ones - y).T, np.log(ones - y_hat))
	const2 = lambda_ / (2 * len(y))
	reg = const2 * l2(theta)
	res = const1 * log_loss_elems + reg
	return res


if __name__=="__main__":
	y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
	y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
	theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
	# Example :
	print(reg_log_loss_(y, y_hat, theta, .5))
	# Output:	0.43377043716475955
	# Example :
	print(reg_log_loss_(y, y_hat, theta, .05))
	# Output:	0.13452043716475953
	# Example :
	print(reg_log_loss_(y, y_hat, theta, .9))
	# Output:	0.6997704371647596

