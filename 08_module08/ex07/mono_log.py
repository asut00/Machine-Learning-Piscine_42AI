# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mono_log.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/20 11:52:46 by asuteau           #+#    #+#              #
#    Updated: 2024/06/20 11:52:48 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alive_progress import alive_bar


class MyLogisticRegression():
	
	def __init__(self, theta, alpha=0.001, max_iter=1000):
		self.theta = theta
		self.alpha = alpha
		self.max_iter = max_iter
		
	def sigmoid_(self, x):
		res = 1 / (1 + np.exp(-x))
		return res

	def add_intercept_(self, x):
		intercept = np.ones((x.shape[0], 1))
		X = np.hstack((intercept, x.reshape(-1, 1) if x.ndim == 1 else x))
		return X

	def predict_(self, x):
		Xprime = self.add_intercept_(x)
		prediction = np.dot(Xprime, self.theta)
		res = self.sigmoid_(prediction)
		return res

	def gradient_(self, x, y, theta):
		Xprime = self.add_intercept_(x)
		XprimeT = Xprime.T
		l_pred = self.predict_(x)
		res = (1/len(x)) * np.dot(XprimeT, (l_pred - y))
		return (res)

	def fit_(self, x, y):
		with alive_bar(self.max_iter, title='Processing') as bar:
			for i in range(self.max_iter):
				self.theta = self.theta - self.alpha * self.gradient_(x, y, self.theta)
				bar()

	def loss_(self, y, y_hat, eps=1e-15):
		elems = y * np.log(y_hat + eps) + ((1 - y) * np.log(1 - y_hat + eps))
		sum = np.sum(elems)
		res = -(1/len(y)) * sum
		return res
	

def data_splitter(x, y, proportion):

	# permutation = np.random.permutation(len(x))

	# x_shf = x[permutation]
	# y_shf = y[permutation]

	x_shf = x
	y_shf = y

	len_train = int(len(x) * proportion)

	x_train = x_shf[:len_train]
	y_train = y_shf[:len_train]
	x_test = x_shf[len_train:]
	y_test = y_shf[len_train:]

	return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy().reshape(-1,1), y_test.to_numpy().reshape(-1,1)



def program(zipcode):
	data = pd.read_csv("./solar_system_census.csv")
	origin_data = pd.read_csv("./solar_system_census_planets.csv")

	# on clean la data
	data = data.drop(data.columns[0], axis=1)
	# origin_data = origin_data.iloc[:, 1]
	origin_data = origin_data.drop(origin_data.columns[0], axis=1)

	beltpeople = (origin_data["Origin"] == 3).astype(int)

	print(type(data))
	print(type(beltpeople))

	x_train, x_test, y_train, y_test = data_splitter(data, beltpeople, 0.8)

	MyLR = MyLogisticRegression

	thetas = np.array([[2], [0.5], [7.1], [4.3]]).reshape(-1, 1)
	
	mylogr = MyLR(thetas, 0.005, 100000)

	print("*****\nOrginaly :")
	p = mylogr.predict_(x_train)

	print(f"prediction shape is : \n{p.shape}\n")
	l = mylogr.loss_(y_train ,p)
	print(f"loss is :\n{l}")


	print()
	print("*****\nAfter prediction :")

	mylogr.fit_(x_train, y_train)
	print(f"thetas are : \n{mylogr.theta}")

	pred = mylogr.predict_(x_test)

	# print(f"prediction is : \n{pred}")
	# print(f"prediction shape is : \n{pred.shape}")

	# print(f"target shape is : \n{y_test.shape}")


	# print(f"y_test shape is {y_test.shape}")
	# print(f"pred shape is {pred.shape}")


	# exit(1)

	l = mylogr.loss_(y_test, pred)
	print(f"loss is : \n{l}")


	# plt.scatter(x_test[:,0], y_test, label='Data points')
	# plt.scatter(x_test[:,0], pred, label='Prediction points')
	# plt.legend()
	# plt.show()

	# print(f"y test shape is {y_test.shape}")
	# print(f"pred shape is {pred.shape}")

	final = np.hstack((y_test, pred))

	# for l in range(len(final)):


	# print(np.round(final, 2))

	# exit(1)






if __name__=="__main__":
	program(0)








######################3



	# second_column = origin_data.iloc[:, 1]
	
	# data['planet'] = second_column.values





	# MyLR = MyLogisticRegression

	# thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
	
	# mylr = MyLR(thetas)

	# print("Orginaly :")
	# p = mylr.predict_(X)
	# print(f"prediction is : \n{p}")
	# l = mylr.loss_(Y,p)
	# print(f"loss is {l}")



	# print("After prediction :")

	# mylr.fit_(X, Y)
	# print(f"thetas are : \n{mylr.theta}")

	# pred = mylr.predict_(X)
	# print(f"prediction is : \n{pred}")
	
	# l = mylr.loss_(Y,pred)
	# print(f"loss is : \n{l}")