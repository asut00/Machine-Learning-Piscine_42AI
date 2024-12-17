# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    multivariate_linear_model.py                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/19 12:27:29 by asuteau           #+#    #+#              #
#    Updated: 2024/06/19 12:27:30 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import numpy as np
from mylinearregression import MyLinearRegression
import pandas as pd

if __name__=="__main__":
	MyLR = MyLinearRegression

	data = pd.read_csv("./spacecraft_data.csv")

	# # Train with thrust age :
	Xage = np.array(data['Age']).reshape(-1,1)
	# Ysellp = np.array(data["Sell_price"]).reshape(-1,1)
	# myLR_age = MyLR(thetas = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
	# myLR_age.fit_(Xage, Ysellp)
	# y_pred = myLR_age.predict_(Xage)
	# print(f"myLR_age.mse_(Y, y_pred) is {myLR_age.mse_(Ysellp, y_pred)}")
	# # Output 55736.867198...
	# myLR_age.plot(Xage, Ysellp, y_pred)

	# # Train with thrust power :
	Xtp = np.array(data['Thrust_power']).reshape(-1,1)
	# Ysellp = np.array(data["Sell_price"]).reshape(-1,1)
	# myLR_thrust = MyLR(thetas = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
	# myLR_thrust.fit_(Xtp, Ysellp)
	# y_pred = myLR_thrust.predict_(Xtp)
	# print(f"myLR_age.mse_(Y, y_pred) is {myLR_thrust.mse_(Ysellp, y_pred)}")
	# myLR_thrust.plot(Xtp, Ysellp, y_pred)

	# # Train with distance :
	Xdist = np.array(data['Terameters']).reshape(-1,1)
	# Ysellp = np.array(data["Sell_price"]).reshape(-1,1)
	# myLR_dist = MyLR(thetas = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
	# myLR_dist.fit_(Xdist, Ysellp)
	# y_pred = myLR_dist.predict_(Xdist)
	# print(f"myLR_age.mse_(Y, y_pred) is {myLR_dist.mse_(Ysellp, y_pred)}")
	# myLR_dist.plot(Xdist, Ysellp, y_pred)


	Xfull = np.array(data[['Age','Thrust_power','Terameters']])
	Ysellp = np.array(data["Sell_price"])

	my_lreg = MyLR(thetas=[1.0, 1.0, 1.0, 1.0], alpha=9e-5, max_iter=500000)
	print(my_lreg.mse_(Ysellp, my_lreg.predict_(Xfull)))

	print("fitting results...")
	my_lreg.fit_(Xfull, Ysellp)
	print(f"my_lreg.thetas is '{my_lreg.thetas}'")

	print(f"print(my_lreg.mse_(Y, my_lreg.predict_(X))) is {my_lreg.mse_(Ysellp, my_lreg.predict_(Xfull))}")

	print()
	
	y_pred = my_lreg.predict_(Xfull)

	my_lreg.scatter(Xage, Ysellp, y_pred)

	my_lreg.scatter(Xtp, Ysellp, y_pred)

	my_lreg.scatter(Xdist, Ysellp, y_pred)

	