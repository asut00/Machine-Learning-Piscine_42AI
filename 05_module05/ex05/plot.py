# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/17 15:56:28 by asuteau           #+#    #+#              #
#    Updated: 2024/06/17 15:56:29 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, theta):
	y_pred = theta[0] + theta[1] * x
	plt.scatter(x, y, color='blue', label='Data points')
	plt.plot(x, y_pred, color='red', label='Prediction line')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.show()

if __name__=="__main__":
	import numpy as np
	x = np.arange(1,6)
	y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
	# Example 1:
	theta1 = np.array([[4.5],[-0.2]])
	# plot(x, y, theta1)

	# Example 2:
	theta2 = np.array([[3.75],[0.2]])
	plot(x, y, theta2)

	# Example 3:
	theta3 = np.array([[3],[0.3]])
	# plot(x, y, theta3)