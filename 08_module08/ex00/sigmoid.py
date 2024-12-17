# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sigmoid.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/20 09:54:46 by asuteau           #+#    #+#              #
#    Updated: 2024/06/20 09:54:47 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math

def sigmoid_(x):


	res = 1 / (1 + np.exp(-x))
	return res

if __name__=="__main__":

	# Example 1:
	x = np.array([[-4]])
	print(sigmoid_(x))
	# Output: array([[0.01798620996209156]])

	# Example 2:
	x = np.array([[2]])
	print(sigmoid_(x))
	# Output: array([[0.8807970779778823]])

	# Example 3:
	x = np.array([[-4], [2], [0]])
	print(sigmoid_(x))
	# Output: array([[0.01798620996209156], [0.8807970779778823], [0.5]])