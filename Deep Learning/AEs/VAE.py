import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy

class DenseLayer(tf.keras.layers.Layer):
	def __init__(self, M1, M2, f=tf.nn.relu):
		super(DenseLayer, self).__init__()
		self.W = tf.Variable(tf.random.normal([M1, M2])* 2/np.sqrt(M2), trainable=True)
		self.b = tf.Variable(tf.zeros([M2], tf.float32), trainable=True)
		self.f = f

	def forward(self, X):
		return self.f(tf.matmul(X, self.W) +self.b)

class VariationalAutoEncoder(tf.keras.layers.Layer):
	def __init__(self, D, hidden_layer_sizes):
		super(VariationalAutoEncoder, self).__init__()
		self.encoder_hidden_layers = []
		self.decoder_hidden_layers = []
		self.M = hidden_layer_sizes[-1]
		self.X = None
		self.mean = None
		self.std = None
		self.Z = None
		self.x_hat_distribution = None
		self.posterior_probs = None

		#Encoder Layers
		M1 = D
		for M2 in hidden_layer_sizes[:-1]:
			layer = DenseLayer(M1, M2)
			self.encoder_hidden_layers.append(layer)
			M1 = M2
		M2 = hidden_layer_sizes[-1]
		layer = DenseLayer(M1, 2*M2, f=lambda x: x)
		self.encoder_hidden_layers.append(layer)

		#Decoder Layers
		M1 = self.M
		for M2 in reversed(hidden_layer_sizes[:-1]):
			layer = DenseLayer(M1, M2)
			self.decoder_hidden_layers.append(layer)
			M1 = M2
		M2 = D
		layer = DenseLayer(M1, M2, f=lambda x: x)
		self.decoder_hidden_layers.append(layer)

	def encoder_forward(self, X):
		self.X = X
		output = X
		for layer in self.encoder_hidden_layers:
			output = layer.forward(output)
		self.mean =  output[:,:(self.M)]
		self.std =  tf.nn.softplus(output[:,(self.M):]) + 1e-6
		normal = tf.compat.v1.distributions.Normal(
			loc=np.zeros(self.M, np.float32),
			scale=np.ones(self.M, np.float32))
		eps = normal.sample(tf.shape(self.mean)[0])
		self.Z = self.mean + eps*self.std

	def decoder_forward(self):
		#posterior
		output = self.Z
		for layer in self.decoder_hidden_layers:
			output = layer.forward(output)
		posterior_logits = output
		self.x_hat_distribution = tf.compat.v1.distributions.Bernoulli(logits=posterior_logits)
		self.posterior_probs = tf.nn.sigmoid(posterior_logits)

	def forward(self, X):
		self.encoder_forward(X)
		self.decoder_forward()

	def cost(self, X):
		# KL divergence
		KL = -tf.math.log(self.std) + 0.5*(self.mean**2 + self.std**2) - 0.5
		KL = tf.math.reduce_sum(KL, axis=1)
		#Expected Log Likelihood 
		E_log_p = tf.math.reduce_sum(self.x_hat_distribution.log_prob(X))
		return -tf.math.reduce_sum(E_log_p - KL)

	def gradient_update(self, X, optimizer):
		with tf.GradientTape() as t:
			self.forward(X)
			Loss = self.cost(X)
		grads = t.gradient(Loss, self.trainable_weights)
		optimizer.apply_gradients(zip(grads, self.trainable_weights))
		return Loss

	def fit(self, X, epochs=20, batch_size=1000, lr=0.005):
		N = X.shape[0]
		n_batches = N // batch_size
		print('Train data........')
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
		cost_lst = []
		for i in range(epochs):
			np.random.shuffle(X)
			for j in range(n_batches):
				Loss = self.gradient_update(X[(j*batch_size):((j+1)*batch_size)], optimizer)
				cost_lst.append(Loss/batch_size)
				if j % 10 ==0:
					print(f'Epoch: {i+1}, Batch: {j}, Loss: {Loss/batch_size}')
		return cost_lst

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
	X = np.concatenate((x_train, x_test), axis=0)
	
	model = VariationalAutoEncoder(D, [200, 100, 2])
	cost_lst = model.fit(X)
	#Loss Curve
	plt.plot(cost_lst)
	plt.title('Loss Curve')
	plt.show()

	#latent space
	row, col = 10, 10
	x = np.linspace(-100, 100,row)
	[x,y] = np.meshgrid(x,x)
	model.Z = (np.array([x.flatten(),tf.nn.softplus(y.flatten())]).T).astype(np.float32)
	model.decoder_forward()
	x_hat = model.posterior_probs
	x_hat = x_hat.numpy()
	fig, axs = plt.subplots(row,col)
	idx = 0
	for i in range(row):
		for j in range(col):
			axs[i,j].imshow(x_hat[idx,:].reshape(28,28), cmap='gray')
			axs[i,j].axis('off')
			idx += 1

	plt.show()