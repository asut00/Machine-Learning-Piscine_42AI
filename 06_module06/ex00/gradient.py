# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    gradient.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/18 09:38:50 by asuteau           #+#    #+#              #
#    Updated: 2024/06/18 09:38:51 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np



def simple_gradient(x, y, theta):
	loss_elems = theta[0] + theta[1] * x
	# z = np.ones((1, 5))
	# # print(x.shape)
	# print(f"x is {x}")
	# print(f"z is {z}")
	# print(f"x*z is {x*z}")
	# print(f"x dot z is {np.dot(z,x)}")
	# print (loss_elems)
	nabla0_diff_sum = np.sum(loss_elems - y)
	nabla0 = nabla0_diff_sum / len(x)
	# print(nabla0)
	nabla1_diff_sum = np.sum((loss_elems - y) * x)
	nabla1 = nabla1_diff_sum / len(x)
	# print(nabla1)
	l = [nabla0, nabla1]
	return (np.array(l))



if __name__=="__main__":
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

	# print(x)
	# print(y)

	# Example 0:
	theta1 = np.array([2, 0.7]).reshape((-1, 1))
	# print(theta1)
	print(simple_gradient(x, y, theta1))
	
	# Example 1:
	theta2 = np.array([1, -0.4]).reshape((-1, 1))
	print(simple_gradient(x, y, theta2))
	# Output: array([[-57.86823748], [-2230.12297889]])