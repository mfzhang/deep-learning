""" Filename: crbm.py """
import numpy as np
import os
import Image   
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams

def readFile(file_name):
	f = np.fromfile(file_name, dtype = np.uint8)
	image = np.array(f, dtype = np.uint8).reshape((5000, 3, 96*96))	
	return image
"""	for k in range(0,200):
		b =[]
		for i in range(0,96):
			a = []
			for j in range(0,96):
				a.append([image[k][0][j*96+i], image[k][1][j*96+i], image[k][2][j*96+i]])
			b.append(a)
		c = np.asarray(b, dtype = np.uint8)
		img = Image.fromarray(c,'RGB')
		img.save(str(k) +'.png')
"""	
	
class CRBM(object):
	"""convolutional Restricted Boltzmann Machine """
	def _init_(self, input = None, input_channel = None, input_size = None, filter_channel = None \
		filter_size = None, pooling_size = None, W = None, vbias = None, hbias = None,  \
		theano_rng = None):
		
		self.input_channel = input_channel
		self.input_size = input_size
		self.filter_channel = filter_channel
		self.filter_size = filter_size
		self.pooling_size = pooling_size

		if numpy_rng is None:
			numpy_rng = numpy.random.RandomState(1234)
		
		if theano_rng is None:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
		
		if W is None:
			"""初始化权重"""
			initial_W = 0.01*np.random.randn(filter_channel, input_channel, filter_size, filter_size)
			W = theano.shared(value=initial_W, name='W', borrow=True)
		
		if vbias is None:
			vbias = theano.shared(value = numpy.zeros(input_channel, dtype = theano.config.floatX),
									name = 'vbias', borrow = True)
		if hbias is None:
			hbias = theano.shared(value = -0.1*numpy.ones(filter_channel, 
									dtype = theano.config.floatX),
									name = 'hbias', borrow = True)
		
		self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]

	def convolutionForward(self):
		conv_out = conv.conv2d(input = input, filters = self.W, 
							filter_shape= self.filter_size, image_shape = self.input_size)
		return [conv_out, T.nnet.sigmoid(conv_out)]
	
	def sampleHidden(self):
		pre_sigmoid_h1, h1_mean = self.convolutionForward()
		h1_sample =  self.theano_rng.binomial(size=h1_mean.shape,
		 									n=1, p=h1_mean,
		 									dtype=theano.config.floatX)
		return [pre_sigmoid_h1, h1_mean, h1_sample]
	
def main():
	#1.得到输入图片
	layer1_input_image = readFile('../../../../crd/deeplearning/data/stl10/train_X.bin')
	#2.初始化参数
	
	
	
	
if __name__ == "__main__":
	main()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
