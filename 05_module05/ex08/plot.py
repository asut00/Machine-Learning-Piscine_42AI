# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/17 16:46:47 by asuteau           #+#    #+#              #
#    Updated: 2024/06/17 16:46:50 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt

def loss_(y, y_hat):
	# le = loss_elem_(y, y_hat)
	# res = y_hat - y
	# for elem in le:
	# 	res += elem
	le = (y_hat - y) ** 2
	le_sum = np.sum(le)
	return (le_sum / (len(y_hat)))

def plot_with_loss(x, y, theta):
	y_pred = theta[0] + theta[1] * x
	loss = loss_(y, y_pred)
	print(loss)
	plt.scatter(x, y, color='blue', label='Data points')
	plt.plot(x, y_pred, color='red', label='Prediction line')

	for i in range(len(x)):
		plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'r--')
	
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.title(f"Loss: {loss}")
	plt.show()


if __name__=="__main__":
	x = np.arange(1,6)
	y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

	# Example 1:
	theta1= np.array([18,-1])
	plot_with_loss(x, y, theta1)

	# Example 2:
	theta2 = np.array([14, 0])
	plot_with_loss(x, y, theta2)

	# Example 3:
	theta3 = np.array([12, 0.8])
	plot_with_loss(x, y, theta3)