# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    TinyStatistician.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: asuteau <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/12 15:06:15 by asuteau           #+#    #+#              #
#    Updated: 2024/06/12 15:06:17 by asuteau          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import math

class TinyStatistician:
		
	def mean(self, data):
		if not data:
			return None
		total = 0
		for elem in data:
			total += elem
		return (total / len(data))
	
	def median(self, data):
		if not data:
			return None
		data.sort()
		datalen = len(data)
		if (datalen % 2 == 0):
			return ((data[(datalen // 2)- 1] + data[datalen // 2])/2)
		else :
			return (data[int(datalen // 2)])
		
	def quartiles(self, data):
		if not data:
			return None
		data.sort()
		fq = data[len(data) // 4]
		tq = data[3 * len(data) // 4]
		qlist = [fq, tq]
		return (qlist)
	
	def var(self, data):
		if not data:
			return None
		moy = self.mean(data)
		total = 0
		for elem in data:
			total += (elem - moy) ** 2
		return (total / len(data))
	
	def std(self, data):
		if not data:
			return None
		stdvar = math.sqrt(self.var(data))
		return (stdvar)





if __name__=="__main__":
	tstat = TinyStatistician()
	a = [1, 42, 300, 10, 59]

	print(f"mean is : {tstat.mean(a)}")
	print(f"median is : {tstat.median(a)}")
	print(f"quartiles are : {tstat.quartiles(a)}")
	print(f"var is : {tstat.var(a)}")
	print(f"std is : {tstat.std(a)}")










	# def quartiles(self, data):
		# median = self.median(data)
		# data.sort() # necessaire ?
		# lenx = len(data)
		# datalen = len(data)
		
		# # Find Q1
		# if datalen % 2 == 0:
		# 	lower_half = data[:datalen // 2]
		# else:
		# 	lower_half = data[:datalen // 2]
		
		# q1 = self.median(lower_half)
		
		# # Find Q3
		# if datalen % 2 == 0:
		# 	upper_half = data[datalen // 2:]
		# else:
		# 	upper_half = data[datalen // 2 + 1:]
		
		# q3 = self.median(upper_half)
		
		# return [q1, q3]