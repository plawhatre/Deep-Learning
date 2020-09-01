import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import os, glob, imageio, PIL
import time

def Build_Generator():
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.LeakyReLU())
	model.add(tf.keras.layers.Reshape((7, 7, 256)))

	model.add(tf.keras.layers.Conv2DTranspose(128,
		(5, 5),
		strides=(1, 1 ),
		padding='same',
		use_bias=False))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.LeakyReLU())

	model.add(tf.keras.layers.Conv2DTranspose(64,
		(5, 5),
		strides=(2, 2 ),
		padding='same',
		use_bias=False))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.LeakyReLU())

	model.add(tf.keras.layers.Conv2DTranspose(1,
		(5,5),
		strides=(2, 2),
		padding='same',
		use_bias=False))
	return model

def Build_Discriminator():
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
	model.add(tf.keras.layers.LeakyReLU())
	model.add(tf.keras.layers.Dropout(0.3))

	model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
	model.add(tf.keras.layers.LeakyReLU())
	model.add(tf.keras.layers.Dropout(0.3))

	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(1))

	return model

def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

def generator_loss(fake_output):
	loss = cross_entropy(tf.ones_like(fake_output), fake_output)
	return loss

def train_step(images):
	noise = tf.random.normal([Batch_Size, noise_dims])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
		generated_images = generator(noise, training=True)
		real_output = discriminator(images, training=True)
		fake_output = discriminator(generated_images, training=True)

		gen_loss = generator_loss(fake_output)
		dis_loss = discriminator_loss(real_output, fake_output)

	grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
	grad_dis = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
	generator_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(grad_dis, discriminator.trainable_variables))

def train(dataset, epochs):
	for epoch in range(epochs):
		start = time.time()
		for image_batch in dataset:
			train_step(image_batch)
		
		generate_and_save_images(generator, epoch+1, seed)
		if (epoch+1)%15 == 0:
			checkpoint.save(file_prefix=checkpoint_prefix)

		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
	
	generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
	pred = model(test_input, training=False)
	fig = plt.figure(figsize=(4, 4))
	for i in range(pred.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow(pred[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
		plt.axis('off')

	plt.savefig('Results/image_at_epoch_{:04d}.png'.format(epoch))
	# plt.show(block=False)
	plt.close('all')

def display_image(epoch_no):
  return PIL.Image.open('Results/image_at_epoch_{:04d}.png'.format(epoch_no))

if __name__ == '__main__':
	if not os.path.exists('Results'):
		os.mkdir('Results')
	(x_train, y_train),  (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train, x_test = (x_train - (255/2))/ (255/2), (x_test - (255/2))/ (255/2)
	N_train, H, W = np.shape(x_train)
	N_test, H, W = np.shape(x_test)
	x_train = x_train.reshape(N_train, H*W)
	x_test = x_test.reshape(N_test, H*W)
	X = np.concatenate((x_train, x_test), axis=0)
	X = X.reshape(X.shape[0], 28, 28, 1).astype('float32')
	Y = np.concatenate((y_train, y_test))

	Buffer_Size = 70000
	Batch_Size = 256

	#batch and shuffle data
	train_dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(Buffer_Size).batch(Batch_Size)
	
	# build models
	generator = Build_Generator()
	discriminator = Build_Discriminator()

	#loss function
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	#optimizers
	generator_optimizer = tf.keras.optimizers.Adam(1e-4)
	discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

	#save checkpoints
	checkpoint_dir = './Results/training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
	checkpoint = tf.train.Checkpoint(
		generator_optimizer=generator_optimizer,
		discriminator_optimizer=discriminator_optimizer,
		generator=generator,
		discriminator=discriminator)

	# Training Loop
	Epochs = 50
	noise_dims = 100
	ex_to_generate = 16
	seed = tf.random.normal([ex_to_generate, noise_dims])
	train(train_dataset, Epochs)

	#GIF
	display_image(Epochs)
	anim_file = 'Results/dcgan.gif'

	with imageio.get_writer(anim_file, mode='I') as writer:
	  filenames = glob.glob('Results/image*.png')
	  filenames = sorted(filenames)
	  last = -1
	  for i,filename in enumerate(filenames):
	    frame = 2*(i**0.5)
	    if round(frame) > round(last):
	      last = frame
	    else:
	      continue
	    image = imageio.imread(filename)
	    writer.append_data(image)
	  image = imageio.imread(filename)
	  writer.append_data(image)