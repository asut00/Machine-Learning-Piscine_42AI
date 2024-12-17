# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    z_score.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/18 18:27:41 by asuteau           #+#    #+#              #
#    Updated: 2024/06/18 18:27:43 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math

def mean(data):
	# if not data:
	# 	return None
	total = 0
	for elem in data:
		total += elem
	return (total / len(data))

def var(data):
	# if not data:
	# 	return None
	moy = mean(data)
	total = 0
	for elem in data:
		total += (elem - moy) ** 2
	return (total / len(data))

def std(data):
	# if not data:
	# 	return None
	stdvar = math.sqrt(var(data))
	return (stdvar)

def zscore(x):
	res = (x - np.mean(x)) / np.std(x)
	return res


if __name__=="__main__":
	# Example 1:
	X = np.array([0, 15, -9, 7, 12, 3, -21])
	print(zscore(X))
	# Output:	array([-0.08620324, 1.2068453 , -0.86203236, 0.51721942, 0.94823559, 0.17240647, -1.89647119])
	# Example 2:
	Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	print(zscore(Y))
	# Output: array([ 0.11267619, 1.16432067, -1.20187941,  0.37558731, 0.98904659, 0.28795027, -1.72770165])

