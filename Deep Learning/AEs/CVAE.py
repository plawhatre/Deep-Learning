import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy

class DenseLayer(tf.keras.layers.Layer):
	def __init__(self, M1, M2, f=tf.nn.relu):
		super(DenseLayer, self).__init__()
		initializer = tf.keras.initializers.GlorotNormal()
		self.W = tf.Variable(initializer(shape=[M1, M2], dtype = tf.float32), trainable=True)
		self.b = tf.Variable(tf.zeros([M2], tf.float32), trainable=True)
		self.f = f

	def forward(self, X):
		return self.f(tf.matmul(X, self.W) +self.b)

class CVAE(tf.keras.layers.Layer):
	def __init__(self, D, n_class, hidden_layer_sizes):
		super(CVAE, self).__init__()
		self.encoder_hidden_layers = []
		self.decoder_hidden_layers = []
		self.M = hidden_layer_sizes[-1]
		self.X = None
		self.n_class = n_class
		self.mean = None
		self.std = None
		self.Z = None
		self.x_hat_distribution = None
		self.posterior_probs = None
		self.posterior_logits = None

		#Encoder Layers
		M1 = D + self.n_class
		for M2 in hidden_layer_sizes[:-1]:
			layer = DenseLayer(M1, M2)
			self.encoder_hidden_layers.append(layer)
			M1 = M2
		M2 = hidden_layer_sizes[-1]
		layer = DenseLayer(M1, 2*M2, f=lambda x: x)
		self.encoder_hidden_layers.append(layer)

		#Decoder Layers
		M1 = self.M + self.n_class
		for M2 in reversed(hidden_layer_sizes[:-1]):
			layer = DenseLayer(M1, M2)
			self.decoder_hidden_layers.append(layer)
			M1 = M2
		M2 = D
		layer = DenseLayer(M1, M2, f=lambda x: x)
		self.decoder_hidden_layers.append(layer)

	def encoder_forward(self, X, Y):
		self.X = X
		output = np.concatenate((X, Y), axis=1)
		for layer in self.encoder_hidden_layers:
			output = layer.forward(output)
		self.mean =  output[:,:(self.M)]
		self.std =  tf.nn.softplus(output[:,(self.M):]) + 1e-6
		normal = tf.compat.v1.distributions.Normal(
			loc=np.zeros(self.M, np.float32),
			scale=np.ones(self.M, np.float32))
		eps = normal.sample(tf.shape(self.mean)[0])
		# self.Z = self.mean + eps*self.std
		self.Z = self.mean + eps*tf.math.exp(self.std/2)

	def decoder_forward(self, Y):
		#posterior
		output = np.concatenate((self.Z, Y), axis=1)
		for layer in self.decoder_hidden_layers:
			output = layer.forward(output)
		self.posterior_logits = output
		self.x_hat_distribution = tf.compat.v1.distributions.Bernoulli(logits=self.posterior_logits)
		self.posterior_probs = tf.nn.sigmoid(self.posterior_logits)

	def forward(self, X, Y):
		self.encoder_forward(X, Y)
		self.decoder_forward(Y)

	def cost(self, X):
	    # E[log P(X|z,y)]
	    recon_loss = tf.math.reduce_sum(
	    	tf.nn.sigmoid_cross_entropy_with_logits(logits=self.posterior_logits, labels=X), 1)
	    # D_KL(Q(z|X,y) || P(z|X,y))
	    kl_loss = 0.5 * tf.math.reduce_sum(tf.math.exp(self.std) + self.mean**2 - 1. - self.std, 1)
	    # VAE loss
	    vae_loss = tf.reduce_mean(recon_loss + kl_loss)

	    return vae_loss

	def gradient_update(self, X, Y, optimizer):
		with tf.GradientTape() as t:
			X = tf.convert_to_tensor(X, np.float32)
			t.watch(X)
			self.forward(X, Y)
			Loss = self.cost(X)
		grads = t.gradient(Loss, self.trainable_weights)
		optimizer.apply_gradients(zip(grads, self.trainable_weights))
		return Loss

	def fit(self, X, Y, epochs=20, batch_size=64, lr=0.001):
		N = X.shape[0]
		n_batches = N // batch_size
		print('Train data........')
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
		cost_lst = []
		for i in range(epochs):
			np.random.shuffle(X)
			for j in range(n_batches):
				Loss = self.gradient_update(
					X[(j*batch_size):((j+1)*batch_size)],
					Y[(j*batch_size):((j+1)*batch_size)],
					optimizer)
				cost_lst.append(Loss/batch_size)
				if j % 100 ==0:
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
	X = np.concatenate((x_train, x_test), axis=0)
	n_class = np.unique(y_train).size
	y_train = tf.one_hot(y_train, depth=10)
	y_test = tf.one_hot(y_test, depth=10)
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)
	X = np.concatenate((x_train, x_test), axis=0)
	Y = np.concatenate((y_train, y_test), axis=0)
	
	model = CVAE(D, n_class, [128, 100, 2])
	cost_lst = model.fit(X, Y, epochs=100, lr=1e-3)
	#Loss Curve
	plt.plot(cost_lst)
	plt.title('Loss Curve')
	plt.show()

	#latent space
	row, col = 5, 5
	x = np.linspace(-1, 1,row)
	[x,y] = np.meshgrid(x,x)
	model.Z = (np.array([x.flatten(),tf.nn.softplus(y.flatten())]).T).astype(np.float32)
	choice = 1
	num = tf.one_hot(np.array([choice]), depth=10).numpy().reshape(1, - 1)
	num = np.repeat(num, repeats=row*col, axis=0)
	model.decoder_forward(num)
	x_hat = model.posterior_logits
	x_hat = x_hat.numpy()
	fig, axs = plt.subplots(row,col)
	idx = 0
	for i in range(row):
		for j in range(col):
			axs[i,j].imshow(x_hat[idx,:].reshape(28,28), cmap='gray')
			axs[i,j].axis('off')
			idx += 1

	plt.show()