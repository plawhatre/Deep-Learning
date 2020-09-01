import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from colorama import init
from termcolor import *
init()

class AutoEncoder:
	def __init__(self, D, M):
		cprint('AutoEncoder Initilization...........','red')
		# Input ---------> Hidden
		self.W = tf.Variable(tf.random.normal(shape=(D, M))*2 / np.sqrt(M))
		self.b = tf.Variable(tf.zeros([M], tf.float32))
		# Hidden ---------> Output
		self.V = tf.Variable(tf.random.normal(shape=(M, D))* 2 / np.sqrt(D))
		self.c = tf.Variable(tf.zeros([D], tf.float32))

	def forward(self, X):
		Z = tf.nn.sigmoid(tf.matmul(X, self.W) + self.b)
		logits  = tf.matmul(Z, self.V) + self.c
		X_hat = tf.nn.sigmoid(logits)
		return X_hat, logits, Z

	# Cost
	def cost(self, X, logits):
		loss = tf.math.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits))
		return loss

	def gradient_update(self, X, optimizer):
		with tf.GradientTape() as t:
			Loss = self.cost(X, self.forward(X)[1])

		grads = t.gradient(Loss, [model.W, model.b, model.V, model.c])
		optimizer.apply_gradients(zip(grads, [model.W, model.b, model.V, model.c]))
		return Loss

	def fit(self, X, epochs=10, batch_size=1000, lr=0.005):
		N = X.shape[0]
		n_batches = N // batch_size
		cprint('Train model........','green')
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
		cost_lst = []
		for i in range(epochs):
			np.random.shuffle(X)
			for j in range(n_batches):
				Loss = self.gradient_update(X[(j*batch_size):((j+1)*batch_size)], optimizer)
				cost_lst.append(Loss/batch_size)
				if j % 10 ==0:
					cprint(f'Epoch: {i+1}, Batch: {j}, Loss: {Loss}', 'green')
		return cost_lst

	def predict(self, X):
		cprint('Making Predictions........','blue')
		return self.forward(X)[0]

if __name__ == "__main__":
	(x_train, y_train),  (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train, x_test = x_train / 255, x_test / 255
	N_train, H, W = np.shape(x_train)
	N_test, H, W = np.shape(x_test)
	x_train = x_train.reshape(N_train, H*W)
	x_test = x_test.reshape(N_test, H*W)
	D = H*W
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)
	M = 300
	model = AutoEncoder(D, M)
	cost_lst = model.fit(x_train)
	#make prediction for random image
	i = np.random.choice(N_test)
	X_hat = model.predict(x_test)
	plt.figure
	plt.subplot(121)
	plt.imshow(x_test[i].reshape(28,28), cmap='gray')
	plt.title('Original')
	plt.subplot(122)
	plt.imshow(X_hat[i].numpy().reshape(28,28), cmap='gray')
	plt.title('Reconstructed')
	plt.show()
	plt.plot(cost_lst)
	plt.title('Loss Curve')
	plt.show()


