# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    12_reg.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/21 11:34:14 by asuteau           #+#    #+#              #
#    Updated: 2024/06/21 11:34:15 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np

def iterative_l2(theta):
	sum = 0
	for l in range(1, len(theta)) :
		sum += (theta[l]) ** 2
	return sum

def l2(theta):
	theta_copy = theta.copy()
	theta_copy[0, 0] = 0
	regularization = np.dot(theta_copy.T, theta_copy)
	return float(regularization[0, 0])

if __name__=="__main__":
	x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

	# Example 1:
	print(iterative_l2(x))
	# Output:	911.0
	# Example 2:
	print(l2(x))
	# Output:	911.0
	y = np.array([3,0.5,-6]).reshape((-1, 1))
	# Example 3:
	print(iterative_l2(y))
	# Output:	36.25
	# Example 4:
	print(l2(y))
	# Output: 36.25