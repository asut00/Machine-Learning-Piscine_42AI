# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    prediction.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/17 13:11:57 by asuteau           #+#    #+#              #
#    Updated: 2024/06/17 13:11:58 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def simple_predict(x, theta):
		if len(x.shape) > 1 and x.shape[1] != 1:
			raise TypeError("Vector x must be one dimensional")
		
		res = np.zeros(x.shape)
		for i in range(x.shape[0]):
			res[i] = theta[0] + x[i] * theta[1]
		print(res)
		



if __name__=="__main__":
	x = np.arange(1, 6)
	# [1 2 3 4 5]

	# example 1:
	theta1 = np.array([5, 0])
	simple_predict(x, theta1)

	# Example 2:
	theta2 = np.array([0, 1])
	simple_predict(x, theta2)
	# Output: array([1., 2., 3., 4., 5.])

	# Example 3:
	theta3 = np.array([5, 3])
	simple_predict(x, theta3)
	# Output: array([ 8., 11., 14., 17., 20.])

	# Example 4:
	theta4 = np.array([-3, 1])
	simple_predict(x, theta4)
	# Output: array([-2., -1., 0., 1., 2.])