import tensorflow as tf
import numpy as np
import os


#os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#get file path and labels
def get_files(file_dir):
	#file_dir: file path
	#return image and labels

	cats = []
	labels_cats = []
	dogs = []
	labels_dogs = []

	#load image path and get labels
	for file in os.listdir(file_dir):
		name = file.split('.')
		if name[0] == 'cat':
			cats.append(file_dir  + file)
			labels_cats.append(0)
		else:
			dogs.append(file_dir + file)
			labels_dogs.append(1)

	print ('there are %d cats and %d dogs!\n')%(len(cats), len(dogs))

	#shuffle the images
	image_list = np.hstack((cats,dogs))
	label_list = np.hstack((labels_cats, labels_dogs))
	temp = np.array([image_list, label_list])
	temp = temp.transpose()	#invert
	np.random.shuffle(temp)

	image_list = list(temp[:,0])
	label_list = list(temp[:,1])
	label_list = [int(i) for i in label_list]

	return image_list, label_list


#batch produce
def get_batch(image, label, image_W, image_H, batch_size, capacity):
	'''
	Args:
		image: list type
		label: list type
		image_W: image width
		image_H: image height
		batch_size: batch size
		capacity: the maximum elements in queue
	return:
		image_batch: 4D tensor batch_size * image_W * image_H * 3, dtype = tf.float32
		label_batch: 1D tensor batch_size, dtype = tf.float32
	'''
	image = tf.cast(image, tf.string)
	label = tf.cast(label, tf.int32)

	#make an input queue
	input_queue = tf.train.slice_input_producer([image, label])

	label = input_queue[1]
	image_contents = tf.read_file(input_queue[0])
	image = tf.image.decode_jpeg(image_contents, channels=3)

	image = tf.image.resize_image_with_crop_or_pad(image, image_H, image_W)
        image = tf.image.per_image_standardization(image)

	image_batch, label_batch = tf.train.batch([image,label],
						batch_size = batch_size,
				         	num_threads = 16,
						capacity = capacity
						)

	return image_batch, label_batch

#Test
'''
import matplotlib.pyplot as plt

BATCH_SIZE = 16
CAPACITY = 256
IMG_W = 208
IMG_H = 208

train_dir = '../data/train/'
image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
	i = 0;
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)

	try:
		while not coord.should_stop() and i < 1:
			img, label = sess.run([image_batch, label_batch])

			for j in np.arange(BATCH_SIZE):
				print 'labels:%d' % label[j]
				plt.imshow(img[j, :, :, :])
				plt.show()
			i = i + 1
	except tf.errors.OutOfRangeError:
		print 'done!'
	finally:
		coord.request_stop()
	coord.join(threads)
'''	








