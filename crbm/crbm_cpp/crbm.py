""" Filename: crbm.py """
import numpy as np
import os
import Image   


def readFile(file_name):
	f = np.fromfile(file_name, dtype = np.uint8)
	image = np.array(f, dtype = np.uint8).reshape((5000, 3, 96*96))	
	a =[]
	for i in (1,96*96):
		a.append([image[0][0][i-1], image[0][1][i-1], image[0][2][i-1]])
	b = np.asarray(a)
	img = Image.fromarray(b, 'RGB')
	img.save('1.png')
	print image
	
	
def main():
	readFile('../../../../crd/deeplearning/data/stl10/train_X.bin')
	
if __name__ == "__main__":
	main()
