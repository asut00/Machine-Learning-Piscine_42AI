# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    matrix.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/17 09:33:55 by asuteau           #+#    #+#              #
#    Updated: 2024/06/17 09:33:56 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class Matrix:
	def __init__(self, data=None, shape=None):
		if data is None:
			self.data = np.zeros(shape)
		self.data = np.array(data)
		if shape == None:
			self.shape = np.array(self.data).shape
		self.shape = self.data.shape

	def __mul__(self, other):
		if isinstance(other, Vector):
			if self.shape[1] != other.shape[0]:
				raise ValueError("Matrix columns must equal vector rows for multiplication")

			res = np.zeros((self.shape[0], 1))
			for i in range(self.shape[0]): # pour chaque ligne
				for j in range(self.shape[1]): # pour chaque element dans la ligne (donc colonne)
					res[i] += self.data[i][j] * other.data[j]

			return Vector(res)
		
		elif isinstance(other, Matrix):
			if self.shape[1] != other.shape[0]:
				raise ValueError("Second matrix's columns must equal first matrix rows for multiplication")
			elif self.shape[0] != other.shape[1]:
				raise ValueError("Second matrix's rows must equal first matrix rows for multiplication")
			
			res = np.zeros((self.shape[0], other.shape[1])) # qui sont en realite la mm valeur
			for x in range(other.shape[1]): # pour chaque colonne de other
				for i in range(self.shape[0]): # pour chaque ligne de self
					for j in range(self.shape[1]): # pour chaque element dans la ligne (donc colonne) de self
						res[i][x] += self.data[i][j] * other.data[j][x]
			
			return Matrix(res)
		
	def __repr__(self):
		return f"Matrix\n{self.data}\n{self.shape}"




class Vector(Matrix):
	def __init__(self, data=None, shape=None):
		super().__init__(data,shape)


		if len(self.shape) > 1 and self.shape[1] != 1 :
			raise ValueError("Error: vector must be one dimensional")
	
	def __repr__(self):
		return f"Vector\n{self.data}\n{self.shape}"



if __name__=="__main__":

	# m1 = Matrix([	[0.0, 1.0, 2.0, 3.0],
	# 				[0.0, 2.0, 4.0, 6.0]])
	# m2 = Matrix([	[0.0, 1.0],
	# 				[2.0, 3.0],
	# 				[4.0, 5.0],
	# 				[6.0, 7.0]])
	
	# print(m1 * m2)



	m1 = Matrix([	[0.0, 1.0, 2.0],
			  		[0.0, 2.0, 4.0]])
	v1 = Vector([	[1],
			  		[2],
					[3]])
	# print(m1.shape)
	# print(v1.shape)

	print(m1 * v1)






	######################################





		# if other.shape[1] < 1 :
		# 	i = 0
		# 	for row in self.data:
		# 		print(f"row is {row}")
		# 		for elem in row:
		# 			print(f"elem is {elem}")
		# 			for o in other:
		# 				print(f"o is {o}")
		# 				print(f"elem * o is {elem * o}")
		# 				res[i] += elem * o
		# 		i += 1
		# 	print(res)
		# 	return res