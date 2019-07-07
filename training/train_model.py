import tensorflow as tf
import cv2
import os
import csv
import numpy as np
import argparse
# import urllib.request
import sys

from google.cloud import storage

client = storage.Client()
bucket = client.get_bucket('recipes-with-images')

# parser = argparse.ArgumentParser(description="Training")

# parser.add_argument('--images-path', '-i', dest="images")
# parser.add_argument('--one-hot-data-path', '-d', dest="data")

# args = parser.parse_args()
# args = vars(args)

# args = {
# 	'images': 'gs://recipes-with-images/images/',
# 	'data': 'gs://recipes-with-images/data_one_hot.csv'
# }


disc_dropout = 0.5
gen_dropout = 0.5


def addConvLayer(model, number_of_kernels, input_shape=None, activation='leaky_relu', dropout_rate=None):
	if input_shape:
		model.add(tf.keras.layers.Conv2D(number_of_kernels, kernel_size=3, input_shape=input_shape, padding='same'))
	else:
		model.add(tf.keras.layers.Conv2D(number_of_kernels, kernel_size=3, strides=2, padding='same'))
	model.add(tf.keras.layers.BatchNormalization())
	# model.add(tf.keras.layers.Activation(activation))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.02))
	if dropout_rate:
		model.add(tf.keras.layers.Dropout(rate=dropout_rate))


def addDenseLayer(model, number_nodes, activation='leaky_relu', dropout_rate=None):
	model.add(tf.keras.layers.Dense(number_nodes))
	# model.add(tf.keras.layers.Activation(activation))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.02))
	if dropout_rate:
		model.add(tf.keras.layers.Dropout(rate=dropout_rate))


def discriminator(number_output):
	model = tf.keras.models.Sequential()

	addConvLayer(model, 8, input_shape=(300, 300, 3), dropout_rate=disc_dropout)
	addConvLayer(model, 16, dropout_rate=disc_dropout)
	addConvLayer(model, 32, dropout_rate=disc_dropout)
	addConvLayer(model, 64, dropout_rate=disc_dropout)
	addConvLayer(model, 128, dropout_rate=disc_dropout)
	addConvLayer(model, 256, dropout_rate=disc_dropout)

	model.add(tf.keras.layers.Flatten())

	addDenseLayer(model, 500, dropout_rate=disc_dropout)

	model.add(tf.keras.layers.Dense(number_output))
	model.add(tf.keras.layers.Activation('sigmoid')) # TODO: Which activation to use? Multiple values should be set to 1, not just one

	return model


def addDeconvolutionLayer(model, number_kernels, kernel_size, upsample_size):
	model.add(tf.keras.layers.UpSampling2D(upsample_size))
	model.add(tf.keras.layers.Conv2DTranspose(number_kernels, kernel_size, padding='same'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation('relu'))


def generator(number_input):
	model = tf.keras.models.Sequential()

	model.add(tf.keras.layers.Dense(5*5*256, input_dim=number_input))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation('relu'))
	model.add(tf.keras.layers.Reshape((5, 5, 256)))
	model.add(tf.keras.layers.Dropout(gen_dropout))

	# TODO: upsampling correctly?
	addDeconvolutionLayer(model, 128, 5, 5) # 5 -> 25
	addDeconvolutionLayer(model, 64, 5, 3) # 25 > 75
	addDeconvolutionLayer(model, 32, 5, 2) # 75 > 150
	addDeconvolutionLayer(model, 16, 5, 2) # 150 > 300

	model.add(tf.keras.layers.Conv2DTranspose(3, 5, padding='same'))
	model.add(tf.keras.layers.Activation('tanh'))

	return model


def train(gen_model, disc_model, adv_model, x_train, y_train, number_of_categories, number_of_ingredients):
	number_epochs = 10000
	batch_size = 64

	disc_loss = np.array([0., 0.])
	adv_loss = np.array([0., 0.])

	for i in range(number_epochs):
		indices = np.random.randint(0, x_train.shape[0], size=batch_size)
		real_images = x_train[indices]  # TODO: Need to get the training images somehow
		real_y = y_train[indices, :]

		noise = np.random.uniform(-1., 1., size=[batch_size, 50])
		noise = np.concatenate((noise, real_y), axis=1)

		fake_images = gen_model.predict(noise)
		# fake_images = fake_images[:, :, :, :]

		x = np.concatenate((real_images, fake_images)).reshape(-1, 300, 300, 3)

		# print("number of categories", number_of_categories)
		# print("number_of_ingredients", number_of_ingredients)
		fake_y = np.vstack((np.zeros((number_of_categories + number_of_ingredients, batch_size)), np.ones((1, batch_size))))

		# print("shapes", real_y.T.shape, fake_y.shape)
		y = np.hstack((real_y.T, fake_y))
		# print("y shape", y.shape)

		disc_model.trainable = True
		disc_loss += np.array(disc_model.train_on_batch(x, y.T))

		random_numbers_categories = np.random.randint(0, number_of_categories, size=batch_size)
		random_numbers_ingredients = np.random.randint(0, 2, size=(number_of_ingredients, batch_size))

		adv_y = np.zeros((number_of_categories, batch_size))

		for j in range(batch_size):
			adv_y[random_numbers_categories[j], j] = 1

		adv_y = np.vstack((adv_y, random_numbers_ingredients))
		adv_y = np.vstack((adv_y, np.zeros((1, batch_size)))) # Appending 0s because these AREN'T fake

		adv_y = adv_y.T

		noise = np.random.uniform(-1., 1., size=[batch_size, 50])
		noise = np.concatenate((noise, adv_y), axis=1)

		disc_model.trainable = False
		adv_loss += np.array(adv_model.train_on_batch(noise, adv_y))

		if i % 50 == 0:
			print(i, "disc_loss:", (disc_loss / (i + 1)), "adv_loss:", (adv_loss / (i + 1)))
			os.system("mkdir saved_gen_models")
			file_name = "saved_gen_models/" + str(i) + ".h5"
			gen_model.save(file_name)
			os.system("gsutil cp -r saved_gen_models/"+str(i)+".h5 gs://recipes-with-images/saved_gen_models/"+str(i)+".h5")
			# client.download_blob_to_file("saved_gen_models/" + str(i) + ".h5", 'gs://recipes-with-images/saved_gen_models')




		# TODO: need to build random array of options
		# So for the first 19, needs to be just one category chosen. For the next 250, can be random number

def getTrainingDataInSameOrder(x_train, y_train):
	number_training_instances = len(x_train)
	y = np.zeros((len(y_train), y_train[1].shape[0]))
	x = np.zeros((number_training_instances, 300, 300, 3))
	x = [None] * number_training_instances

	# HOW DO I GET AROUND THIS
	# We have
	for i in range(len(x_train)):
		try:
			row_number = x_train[i][0]
			row_x = x_train[i][1]
			row_y = y_train[row_number]

			y[i, :] = row_y.astype(np.uint8)
			# x[row_number, :, :, :] = row_x
			x[i] = row_x
		except:
			print("i", i, len(x))

	x = np.array(x)
	print("shape", x.shape, y.shape)
	return x, y


# def tempDownloadFiles():
# 	# urllib.request.Request('gs://recipes-with-images/')

# def loadTrainingData(indices):
# 	# Indices correspond to image file names
# 	# TODO: Maybe I should load all images at once??
# 	filenames = ["images/"+str(index)+".jpg" for index in indices]
# 	raw_imgs = [tf.read_file(filename) for filename in filenames]
# 	img_tensors = [(tf.image.decode_image(raw_img - 128)) / 128 for raw_img in raw_imgs]

# 	return img_tensors


def loadAllTrainingImages():
	os.system("mkdir tmp")
	os.system("mkdir tmp/images")
	os.system("mkdir tmp/images/images")

	blobs = bucket.list_blobs(prefix='images')
	# print("all training images")

	# i = 0
	for blob in blobs:
		# print("blob", blob)
		# i += 1
		# if i > 2:
		# 	break
		filename = 'tmp/' + str(blob.name)
		with open(filename, 'wb+') as file_to:
			try:
				client.download_blob_to_file(blob, file_to)
			except e:
				sys.stderr.write("error")
				sys.stderr.write(e)

	# print("past downloading")

	# print("im im",os.system("ls tmp/images/images"))
	# print("im", os.system("ls tmp/images"))

	# with open('tmp/images') as directory:
		# client.download_blob_to_file('gs://recipes-with-images/images', directory)

	filenames = os.listdir('tmp/images/')
	# filenames = bucket.list_blobs('/images/')

	os.system("rm -rf tmp/images/images")
	filepaths = [(name[:-4], 'tmp/images/'+str(name)) for name in filenames]
	# print("filepaths", filepaths)
	# cv2.imread(filepaths[0][1])
	# cv2.imshow("im", im)
	# cv2.waitKey(1000)

	raw_imgs = []
	for filepath in filepaths:
		try:
			image = cv2.imread(filepath[1])
			_id = filepath[0]

			if (filepath[0][0] != "." and filepath[0] != 'im' and not isinstance(image, type(None))):
				raw_imgs.append((_id, image))
		except:
			sys.stderr.write("Something went wrong with raw images")

	# raw_imgs = [(filepath[0], cv2.imread(filepath[1])) for filepath in filepaths if (filepath[0][0] != "." and filepath[0] != 'im')]
	# print("raw_imgs", raw_imgs)
	# raw_imgs = [(filepath[0], cv2.cvtColor(cv2.imread(filepath[1]), cv2.COLOR_BGR2RGB)) for filepath in filepaths]
	img_tensors = [(int(raw_img[0]), ((raw_img[1].astype(np.int16) - 128) / 128)) for raw_img in raw_imgs]
	# print("img_tensors", img_tensors)

	return img_tensors


def loadAllTrainingY(filepath):
	y_train = {}

	# with open(filepath, 'r') as input_file:
	with tf.io.gfile.GFile(filepath, 'r') as input_file:
		csv_reader = csv.reader(input_file, delimiter=",")

		for row in csv_reader:
			_id = int(row[0])
			one_hot_encoded = row[1:]
			# print("len one_hot_encoded", len(one_hot_encoded))
			y_train[_id] = np.hstack((np.array(one_hot_encoded), np.array([0])))

	return y_train


def main():
	x_train = loadAllTrainingImages()

	os.system('touch tmp/data_one_hot.csv')
	with open('tmp/data_one_hot.csv', 'wb+') as file_to:
		blobs = bucket.list_blobs(prefix='data_one_hot.csv')
		for blob in blobs:
			client.download_blob_to_file(blob, file_to)
	y_train = loadAllTrainingY('tmp/data_one_hot.csv')

	x_train, y_train = getTrainingDataInSameOrder(x_train, y_train)

	gen_model = generator(y_train[0].shape[0] + 50)
	disc_model = discriminator(19 + 250 + 1)

	discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=4e-4)

	discriminator_model = tf.keras.models.Sequential()
	discriminator_model.add(disc_model)
	discriminator_model.compile(loss='categorical_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

	adversarial_optimizer = tf.keras.optimizers.RMSprop(lr=1e-4)

	adversarial_model = tf.keras.models.Sequential()
	adversarial_model.add(gen_model)
	adversarial_model.add(disc_model)
	adversarial_model.compile(loss='categorical_crossentropy', optimizer=adversarial_optimizer, metrics=['accuracy'])

	train(gen_model, discriminator_model, adversarial_model, x_train, y_train, 19, 250)


if __name__ == "__main__":
	main()	