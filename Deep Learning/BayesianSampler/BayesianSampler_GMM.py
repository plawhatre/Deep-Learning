import numpy as np
from sklearn.mixture import BayesianGaussianMixture as GMM
import matplotlib.pyplot as plt
import tensorflow as tf
class BayesSampling:
	def __init__(self, comp):
		self.comp = comp
	def fit(self, X, Y):
		self.K = len(set(Y))
		self.gaussians = []
		for k in range(self.K):
			print(f'GMM for class {k}')
			Xk = X[Y == k]
			gmm = GMM(n_components=self.comp)
			gmm.fit(Xk)
			self.gaussians.append(gmm)

	def sample_given_y(self, Y):
		gmm = self.gaussians[Y]
		sample, cluster = gmm.sample()
		mean = gmm.means_[cluster[0]]
		return sample.reshape(28, 28), mean.reshape(28, 28)

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
	n_comp = 10
	sampler = BayesSampling(n_comp)
	sampler.fit(X, Y)
	for k in range(sampler.K):
		sample, mean = sampler.sample_given_y(k)

		plt.subplot(1, 2, 1)
		plt.imshow(sample, cmap='gray')
		plt.title(f'Sample of class {k}')
		plt.subplot(1, 2, 2)
		plt.imshow(mean, cmap='gray')
		plt.title(f'Mean of class {k}')
		plt.show()

	sample, _ = sampler.sample()
	plt.imshow(sample, cmap='gray')
	plt.title('Random sample')
	plt.show()

