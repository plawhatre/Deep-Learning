import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

class Dense(tf.keras.layers.Layer):
	def __init__(self, M1, M2, f=lambda x: x):
		super(Dense, self).__init__()
		initializer = tf.keras.initializers.GlorotNormal()
		self.W = tf.Variable(initializer(shape=[M1, M2], dtype=tf.float32), trainable=True)
		self.b = tf.Variable(tf.zeros([M2], tf.float32), trainable=True)
		self.f = f
		# print('Dense Layer created')

	def forward(self, X):
		# print('Dense Pass')
		return self.f(tf.matmul(X, self.W) + self.b)

class BatchNorm(tf.keras.layers.Layer):
	def __init__(self, batch_size, N):
		super(BatchNorm, self).__init__()
		self.gamma = tf.Variable(tf.ones([batch_size, N]), trainable=True)
		self.beta = tf.Variable(tf.zeros([batch_size, N]), trainable=True)
		self.batch_size = batch_size
		self.N = N
		# print('BatchNorm created')

	def forward(self, X):
		row, col = self.batch_size, self.N
		self.mu = tf.constant(np.repeat(np.mean(X.numpy(), axis=0).reshape(1, -1), row, axis=0))
		self.sigma = tf.constant(np.repeat(np.std(X.numpy(), axis=0).reshape(1, -1), row, axis=0))
		scale =  tf.math.divide((X - self.mu), self.sigma)
		# print('BatchNorm Pass')
		return tf.math.multiply(self.gamma, scale) + self.beta

class LeakyReLU:
	def __init__(self, alpha):
		self.alpha = alpha
		# print('LeakyReLU created')

	def forward(self, X):
		# print('LeakyReLU Pass')
		return tf.maximum(self.alpha*X, X)

class GAN(tf.keras.layers.Layer):
	def __init__(self, dim_z, dim_x, batch_size, generator_layer_sizes, discriminator_layer_sizes):
		super(GAN, self).__init__()
		self.dim_z = dim_z
		self.dim_x = dim_x
		self.batch_size = batch_size
		self.generator_layer_sizes = generator_layer_sizes
		self.generator_layers = []
		self.discriminator_layer_sizes = discriminator_layer_sizes
		self.discriminator_layers = []
		self.X = None
		self.X_hat = None
		self.Z = None
		print('GAN Initialized..........')

		#Building Model
		self.generator()
		print('Generator Build Complete..........')
		self.discriminator()
		print('Discriminator Build Complete..........')
		print('GAN Build Complete..........')

	def generator(self):
		M1 = self.dim_z
		for M2 in self.generator_layer_sizes:
			layer = Dense(M1, M2)
			self.generator_layers.append(layer)
			layer = BatchNorm(self.batch_size, M2)
			self.generator_layers.append(layer)
			layer = LeakyReLU(0.001)
			self.generator_layers.append(layer)
			M1 = M2

		layer = Dense(M1, self.dim_x)
		self.generator_layers.append(layer)
		layer = BatchNorm(self.batch_size, self.dim_x)
		self.generator_layers.append(layer)
		layer = LeakyReLU(0.001)
		self.generator_layers.append(layer)

	def discriminator(self):
		M1 = self.dim_x
		for M2 in self.discriminator_layer_sizes:
			layer = Dense(M1, M2)
			self.discriminator_layers.append(layer)
			layer = BatchNorm(self.batch_size, M2)
			self.discriminator_layers.append(layer)
			layer = LeakyReLU(0.001)
			self.discriminator_layers.append(layer)
			M1 = M2

		layer = Dense(M1, 1)
		self.discriminator_layers.append(layer)

	def generator_forward(self):
		self.Z = tf.random.normal(shape=[self.batch_size, self.dim_z])
		output = self.Z
		i = 0
		for layer in self.generator_layers:
			i += 1
			output = layer.forward(output)
		# print('Generator Pass')
		self.X_hat = output

	def discriminator_forward(self, X):
		self.X = X
		output = X
		for layer in self.discriminator_layers:
			output = layer.forward(output)
		# print('Discriminator Pass')
		return output

	def gan_forward(self, X):
		self.generator_forward()
		Y_fake = self.discriminator_forward(self.X_hat)
		Y_real = self.discriminator_forward(self.X)
		return Y_real, Y_fake


	def generator_loss(self, Y_fake):
		return tf.math.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_fake, labels=tf.ones_like(Y_fake)))

	def discriminator_loss(self, Y_real, Y_fake):
		real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_real, labels=tf.ones_like(Y_real))
		fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_fake, labels=tf.zeros_like(Y_fake))
		return tf.math.reduce_sum(real_loss + fake_loss)

	def gradient_update(self, X, optimizer):
		with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
			Y_real, Y_fake = self.gan_forward(X)
			g_loss = self.generator_loss(Y_fake)
			d_loss = self.discriminator_loss(Y_real, Y_fake)
		grad_g = g_tape.gradient(g_loss, self.trainable_weights)
		grad_d = d_tape.gradient(d_loss, self.trainable_weights)
		optimizer.apply_gradients(zip(grad_g,self.trainable_weights))
		optimizer.apply_gradients(zip(grad_d,self.trainable_weights))
		return g_loss, d_loss

	def fit(self, X, epochs=20, lr=0.0001):
		N = X.shape[0]
		n_batches = N // self.batch_size
		print('Training data........')
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
		gcost_lst = []
		dcost_lst = []
		for i in range(epochs):
			np.random.shuffle(X)
			for j in range(n_batches):
				gLoss, dLoss = self.gradient_update(
					X[(j*self.batch_size):((j+1)*self.batch_size)],
					optimizer)
				gcost_lst.append((gLoss/self.batch_size).numpy())
				dcost_lst.append((dLoss/self.batch_size).numpy())
				if j % 10 ==0:
					print(f'Epoch: {i+1}, Batch: {j}, Loss: {(gLoss/self.batch_size).numpy(), (dLoss/self.batch_size).numpy()}')
			self.generate_images('Epoch'+str(i))
		return gcost_lst, dcost_lst

	def generate_images(self, name,seed=1):
		tf.random.set_seed(0)
		if name == 'Final':
			tf.random.set_seed(seed)
		self.generator_forward()
		I = self.X_hat.numpy()
		fig, axs = plt.subplots(5, 5)
		n = 1
		for i in range(5):
			for j in range(5):
				axs[i,j].imshow(I[n,:].reshape(28, 28), cmap='gray')
			axs[i,j].axis('off')
			n += 1
		plt.show()
		plt.savefig('Results/' + name + 'generated_images.png')

if __name__ == '__main__':
	if not os.path.exists('Results'):
		os.mkdir('Results')
	(x_train, _ ),  (x_test, _ ) = tf.keras.datasets.mnist.load_data()
	x_train, x_test = (x_train - (255/2))/ (255/2), (x_test - (255/2))/ (255/2)
	N_train, H, W = np.shape(x_train)
	N_test, H, W = np.shape(x_test)
	x_train = x_train.reshape(N_train, H*W)
	x_test = x_test.reshape(N_test, H*W)
	X = np.concatenate((x_train, x_test), axis=0)
	X = X.reshape(X.shape[0], 28, 28, 1).astype('float32')

	model = GAN(100, 28**2, 32, [300, 400, 500], [300,10])
	glst, dlst = model.fit(X, epochs=50)

	#Generate images
	model.generate_images('Final')
	# Loss Curves
	plt.plot(glist, label='generator_loss')
	plt.plot(dlist, label='discriminator_loss')
	plt.legend()
	plt.show()