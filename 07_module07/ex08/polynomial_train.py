# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    polynomial_train.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/19 15:29:21 by asuteau           #+#    #+#              #
#    Updated: 2024/06/19 15:29:22 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from mylinearregression import MyLinearRegression
import pandas as pd
import matplotlib.pyplot as plt

def add_polynomial_features(x, power):
	res = np.zeros((len(x), power))
	for i in range(len(x)): # pour chaque ligne de x
		for j in range(1, power + 1): # pour chaque power
			# print(f"i is {i}")
			res[i][j - 1] = x[i] ** j
	return res

class MyLinearRegression:

	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.thetas = thetas
		self.alpha = alpha
		self.max_iter = max_iter

	def add_intercept(self, x):
		intercept = np.ones((x.shape[0], 1))
		X = np.hstack((intercept, x.reshape(-1, 1) if x.ndim == 1 else x))
		return X

	def mse_(self, y, y_hat):
		loss_elems = (y_hat - y) ** 2
		mse = np.sum(loss_elems) / len(y_hat)
		return mse
	
	def loss_(self, y, y_hat):
		res = 1 / (2 * len(y)) * np.dot((y_hat - y).T, (y_hat - y))
		return res


	def loss_elem_(self, y, y_hat):
		res = np.zeros(y_hat.shape[0])
		for i in range(y_hat.shape[0]):
			res[i] = (y_hat[i] - y[i]) ** 2
		return res

	def gradient(self, x, y, theta):
		Xprime = self.add_intercept(x)
		XprimeT = Xprime.T
		# print(f"Xprime is {Xprime}")
		# print(f"theta is {theta}")
		res = (1 / len(x)) * np.dot(XprimeT, (np.dot(Xprime, theta) - y))
		return res

	def fit_(self, x, y):
		for i in range(self.max_iter):
			# print(f"gradient is {self.gradient(x, y, self.thetas)}")
			self.thetas = self.thetas - self.alpha * self.gradient(x, y, self.thetas)
		return self.thetas
	
	def predict_(self, x):
		prediction = np.dot(self.add_intercept(x), self.thetas)
		return prediction
	
	def plot(self, x, y, y_pred):
		plt.scatter(x, y, color='blue', label='Data points')
		plt.plot(x, y_pred, color='red', label='Prediction line')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend()
		plt.show()

	def scatter(self, x, y, y_pred):
		plt.scatter(x, y, color='blue', label='Data points')
		plt.scatter(x, y_pred, color='red', label='Prediction points')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend()
		plt.show()



if __name__=="__main__":
	data = pd.read_csv("./are_blue_pills_magics.csv")
	Xpill1 = np.array(data["Micrograms"]).reshape(-1, 1)
	Yscore = np.array(data["Score"]).reshape(-1, 1)

	thetas = [
		np.array([[1], [4]]),
		np.array([[10], [20], [12]]),
		np.array([[1], [1], [1], [1]]),
		np.array([[-20], [160], [-80], [10], [-1]]),
		np.array([[1140], [-1850], [1110], [-305], [40], [-2]]),
		np.array([[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]])
	]

	fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
	fig.suptitle('Polynomial Regression with Different Powers')

	for i, ax in enumerate(axes.flat):
		power = i + 1
		poly_xpill = add_polynomial_features(Xpill1, power)
		mlr = MyLinearRegression(thetas[i], 1e-9)
		mlr.fit_(poly_xpill, Yscore)
		y_pred = mlr.predict_(poly_xpill)

		continuous_x = np.arange(0, 7, 0.01).reshape(-1, 1)
		poly_x = add_polynomial_features(continuous_x, power)
		continuous_y_pred = mlr.predict_(poly_x)

		print(f"continuous_x is {continuous_x.shape}")
		print(f"continuous_y_pred is {continuous_x.shape}")
		ax.scatter(Xpill1, Yscore, color='blue', label='Data points')
		ax.plot(continuous_x, continuous_y_pred, color='red', label='Prediction line')
		ax.set_title(f'Polynomial Power {power}')
		ax.set_xlabel('Micrograms')
		ax.set_ylabel('Score')
		ax.legend()

	plt.tight_layout(rect=[0, 0, 1, 0.96])
	plt.show()




	# #5
	# continuous_x = np.arange(0, 7, 0.01).reshape(-1,1)
	# poly_x = add_polynomial_features(continuous_x, 5)
	# continuous_y_pred5 = mlr5.predict_(poly_x)

	# ax.scatter(Xpill1, Yscore, color='blue', label='Data points')
	# ax.plot(continuous_x, continuous_y_pred5, color='red', label='Prediction line')
	# # plt.show()

	# #6
	# continuous_x = np.arange(0, 7, 0.01).reshape(-1,1)
	# poly_x = add_polynomial_features(continuous_x, 6)
	# continuous_y_pred6 = mlr6.predict_(poly_x)






###################################################################





# 	# MyLR = MyLinearRegression
# 	data = pd.read_csv("./are_blue_pills_magics.csv")

# 	Xpill1 = np.array(data["Micrograms"]).reshape(-1,1)
# 	Yscore = np.array(data["Score"]).reshape(-1,1)

# 	Xpill2 = add_polynomial_features(Xpill1, 2)
# 	Xpill3 = add_polynomial_features(Xpill1, 3)
# 	Xpill4 = add_polynomial_features(Xpill1, 4)
# 	Xpill5 = add_polynomial_features(Xpill1, 5)
# 	Xpill6 = add_polynomial_features(Xpill1, 6)

# 	# print(f"Xpill1 is \n{Xpill1}")
# 	# print(f"Xpill2 is \n{Xpill2}")
# 	# print(f"Xpill3 is \n{Xpill3}")

# 	theta1 = np.array([[1], [4]]).reshape(-1,1)
# 	theta2 = np.array([[10], [20], [12]]).reshape(-1,1)
# 	theta3 = np.array([[-20], [160], [-80], [10]]).reshape(-1,1)
# 	theta4 = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
# 	theta5 = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
# 	theta6 = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)


# 	mse = np.zeros((6,2))
# 	for i in range(6):
# 		mse[i][0] = i + 1


# 	mlr1 = MyLinearRegression(theta1)
# 	mlr1.fit_(Xpill1, Yscore)
# 	y_pred1 = mlr1.predict_(Xpill1)
# 	mse[0][1] = mlr1.mse_(Yscore, y_pred1)
# 	print(f"mse1 is {mse[0]}") # on imprime la mse pour le #1


# 	mlr2 = MyLinearRegression(theta2)
# 	mlr2.fit_(Xpill2, Yscore)
# 	y_pred2 = mlr2.predict_(Xpill2)
# 	mse[1][1] = mlr2.mse_(Yscore, y_pred2)
# 	print(f"mse2 is {mse[1]}") # on imprime la mse pour le #2


# 	mlr3 = MyLinearRegression(theta3, 1e-5)
# 	mlr3.fit_(Xpill3, Yscore)
# 	y_pred3 = mlr3.predict_(Xpill3)
# 	mse[2][1] = mlr3.mse_(Yscore, y_pred3)
# 	print(f"mse3 is {mse[2]}") # on imprime la mse pour le #3

# 	mlr4 = MyLinearRegression(theta4, 1e-9)
# 	mlr4.fit_(Xpill4, Yscore)
# 	y_pred4 = mlr4.predict_(Xpill4)
# 	mse[3][1] = mlr4.mse_(Yscore, y_pred4)
# 	print(f"mse4 is {mse[3]}") # on imprime la mse pour le #4

# 	mlr5 = MyLinearRegression(theta5, 1e-9)
# 	mlr5.fit_(Xpill5, Yscore)
# 	y_pred5 = mlr5.predict_(Xpill5)
# 	mse[4][1] = mlr5.mse_(Yscore, y_pred5)
# 	print(f"mse5 is {mse[4]}") # on imprime la mse pour le #5

# 	mlr6 = MyLinearRegression(theta6, 1e-10)
# 	mlr6.fit_(Xpill6, Yscore)
# 	y_pred6 = mlr6.predict_(Xpill6)
# 	mse[5][1] = mlr6.mse_(Yscore, y_pred6)
# 	print(f"mse5 is {mse[5]}") # on imprime la mse pour le #6

# 	print(mse)

# 	# powers = mse[:, 0]
# 	# mse_values = mse[:, 1]

# 	# plt.bar(powers, mse_values, color='blue')
# 	# plt.xlabel('Power')
# 	# plt.ylabel('Mean Squared Error')
# 	# plt.title('MSE for Polynomial Features of Different Powers')
# 	# plt.show()

# 	# mlr1.plot(Xpill1, Yscore, y_pred1)
# 	# mlr2.plot(Xpill2, Yscore, y_pred2)
# 	# mlr3.plot(Xpill3, Yscore, y_pred3)
# 	# mlr4.plot(Xpill4, Yscore, y_pred4)
# 	# mlr5.plot(Xpill5, Yscore, y_pred5)
# 	# mlr6.plot(Xpill6, Yscore, y_pred6)

# 	#1
# 	ax.scatter(Xpill1, Yscore, color='blue', label='Data points')
# 	ax.plot(Xpill1, y_pred1, color='red', label='Prediction line')
# 	# plt.show()

# 	#2
# 	continuous_x = np.arange(0, 7, 0.01).reshape(-1,1)
# 	poly_x = add_polynomial_features(continuous_x, 2)
# 	continuous_y_pred2 = mlr2.predict_(poly_x)

# 	ax.scatter(Xpill1, Yscore, color='blue', label='Data points')
# 	ax.plot(continuous_x, continuous_y_pred2, color='red', label='Prediction line')
# 	# plt.show()

# 	#3
# 	continuous_x = np.arange(0, 7, 0.01).reshape(-1,1)
# 	poly_x = add_polynomial_features(continuous_x, 3)
# 	continuous_y_pred3 = mlr3.predict_(poly_x)

# 	ax.scatter(Xpill1, Yscore, color='blue', label='Data points')
# 	ax.plot(continuous_x, continuous_y_pred3, color='red', label='Prediction line')
# 	# plt.show()

# 	#4
# 	continuous_x = np.arange(0, 7, 0.01).reshape(-1,1)
# 	poly_x = add_polynomial_features(continuous_x, 4)
# 	continuous_y_pred4 = mlr4.predict_(poly_x)

# 	ax.scatter(Xpill1, Yscore, color='blue', label='Data points')
# 	ax.plot(continuous_x, continuous_y_pred4, color='red', label='Prediction line')
# 	# plt.show()

# 	#5
# 	continuous_x = np.arange(0, 7, 0.01).reshape(-1,1)
# 	poly_x = add_polynomial_features(continuous_x, 5)
# 	continuous_y_pred5 = mlr5.predict_(poly_x)

# 	ax.scatter(Xpill1, Yscore, color='blue', label='Data points')
# 	ax.plot(continuous_x, continuous_y_pred5, color='red', label='Prediction line')
# 	# plt.show()

# 	#6
# 	continuous_x = np.arange(0, 7, 0.01).reshape(-1,1)
# 	poly_x = add_polynomial_features(continuous_x, 6)
# 	continuous_y_pred6 = mlr6.predict_(poly_x)

# 	ax.scatter(Xpill1, Yscore, color='blue', label='Data points')
# 	ax.plot(continuous_x, continuous_y_pred6, color='red', label='Prediction line')
# 	# plt.show()

# plt.show()