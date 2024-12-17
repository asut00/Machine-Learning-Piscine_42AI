# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multi_log.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/20 11:52:56 by asuteau           #+#    #+#              #
#    Updated: 2024/06/20 11:52:57 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
from alive_progress import alive_bar
import matplotlib.pyplot as plt

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

	x_shf = x
	y_shf = y

	len_train = int(len(x) * proportion)

	x_train = x_shf[:len_train]
	y_train = y_shf[:len_train]
	x_test = x_shf[len_train:]
	y_test = y_shf[len_train:]

	return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy().reshape(-1,1), y_test.to_numpy().reshape(-1,1)


if __name__=="__main__":
	data = pd.read_csv("./solar_system_census.csv")
	origin_data = pd.read_csv("./solar_system_census_planets.csv")
	MyLR = MyLogisticRegression

	# on clean la data
	data = data.drop(data.columns[0], axis=1)
	origin_data = origin_data.drop(origin_data.columns[0], axis=1)

	origin_list = ["Venus", "Earth", "Mars", "Belt"]

	# print(origin_data, data)

	thetas = np.array([[2], [0.5], [7.1], [4.3]]).reshape(-1, 1)

	mylogr = MyLR(thetas, 0.005, 500000)

	for i in range(4):
		people = (origin_data["Origin"] == i).astype(int)
		x_train, x_test, y_train, y_test = data_splitter(data, people, 0.8)
		print(f"\n//// Analysing civilisation : {origin_list[i]}")
		print("*****\nOrginaly :")
		p = mylogr.predict_(x_train)
		# print(f"prediction shape is : \n{p.shape}\n")
		l = mylogr.loss_(y_train ,p)
		print(f"loss is :\n{l}")

		print()
		print("*****\nAfter prediction :")
		mylogr.fit_(x_train, y_train)
		print(f"thetas are : \n{mylogr.theta}")
		pred = mylogr.predict_(x_test)
		l = mylogr.loss_(y_test, pred)
		print(f"loss is : \n{l}")
		if i == 0:
			full_pred_data = pred
		else :
			full_pred_data = np.hstack((full_pred_data, pred))
		# print(full_pred_data)
		# if i == 1:
		# 	exit(1)

	civ_repart = np.zeros((len(full_pred_data), 1))

	for i in range(len(full_pred_data)):
		max = 0
		for j in range(len(full_pred_data[i])):
			if full_pred_data[i][j] > max:
				max = full_pred_data[i][j]
				civ_repart[i] = j


	# print(civ_repart)

	xou, x1ou, you, y_test_gen = data_splitter(data, origin_data, 0.8)

	print()

	final_compare = np.hstack((civ_repart, y_test_gen))

	# print(final_compare)

	errors = 0

	for i in range(len(final_compare)):
		if final_compare[i][0] != final_compare[i][1]:
			errors += 1

	print(f"number of errors : {errors}/{len(final_compare)}")

	# print(type(x_test))
	# print(type(y_test_gen))
	# print(x_test[:,0].reshape(-1, 1))
	# print(y_test_gen)

	fig, axs = plt.subplots(1, 3, figsize=(18,6))


	axs[0].scatter(x_test[:,0].reshape(-1, 1), y_test_gen, s=30, label='Data points')
	axs[0].scatter(x_test[:,0].reshape(-1,1), civ_repart, s=10, label='Prediction points')
	axs[0].set_xlabel('Weight')
	axs[0].set_ylabel('Civilisation')
	axs[0].legend()
	# axs[0].show()
	

	axs[1].scatter(x_test[:,1].reshape(-1, 1), y_test_gen, s=30, label='Data points')
	axs[1].scatter(x_test[:,1].reshape(-1,1), civ_repart, s=10, label='Prediction points')
	axs[1].set_xlabel('Height')
	axs[1].set_ylabel('Civilisation')
	axs[1].legend()
	# axs[1].show()

	axs[2].scatter(x_test[:,2].reshape(-1, 1), y_test_gen, s=30, label='Data points')
	axs[2].scatter(x_test[:,2].reshape(-1,1), civ_repart, s=10, label='Prediction points')
	axs[2].set_xlabel('Bone density')
	axs[2].set_ylabel('Civilisation')
	axs[2].legend()
	# axs[2].show()

	plt.tight_layout()
	plt.show()

	# beltpeople = (origin_data["Origin"] == 3).astype(int)

	# x_train, x_test, y_train, y_test = data_splitter(data, beltpeople, 0.8)

	# MyLR = MyLogisticRegression

	# thetas = np.array([[2], [0.5], [7.1], [4.3]]).reshape(-1, 1)
	
	# mylogr = MyLR(thetas, 0.005, 100000)

	# print("*****\nOrginaly :")