import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import tensorflow as tf

class BayesSampling:
	def fit(self, X, Y):
		self.K = len(set(Y))
		self.gaussians = []
		for k in range(self.K):
			print(f'Gaussian distribution for class {k}')
			Xk = X[Y == k]
			mean = Xk.mean(axis=0)
			cov = np.cov(Xk.T)
			g = {'m': mean, 'c': cov}
			self.gaussians.append(g)

	def sample_given_y(self, Y):
		g = self.gaussians[Y]
		return mvn.rvs(mean=g['m'], cov=g['c'])

	def sample(self):
		y = np.random.randint(self.K)
		return self.sample_given_y(y)

if __name__ == '__main__':
	(x_train, y_train),  (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train, x_test = x_train / 255, x_test / 255
	N_train, H, W = np.shape(x_train)
	N_test, H, W = np.shape(x_test)
	x_train = x_train.reshape(N_train, H*W)
	x_test = x_test.reshape(N_test, H*W)
	X = np.concatenate((x_train, x_test), axis=0)
	Y = np.concatenate((y_train, y_test))
	sampler = BayesSampling()
	sampler.fit(X, Y)

	for k in range(sampler.K):
		sample = sampler.sample_given_y(k).reshape(28,28)
		mean = sampler.gaussians[k]['m'].reshape(28,28)

		plt.subplot(1, 2, 1)
		plt.imshow(sample, cmap='gray')
		plt.title(f'Sample of class {k}')
		plt.subplot(1, 2, 2)
		plt.imshow(mean, cmap='gray')
		plt.title(f'Mean of class {k}')
		plt.show()

	sample = sampler.sample().reshape(28, 28)
	plt.imshow(sample, cmap='gray')
	plt.title('Random sample')
	plt.show()
	