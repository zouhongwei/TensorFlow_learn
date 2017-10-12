import tensorflow as tf
import os
import numpy as np

def read_files(file_dir, is_train = True, batchsize = 16, is_shuffle = True):
	'''read the data and return image & label batch 
	Args:
		file_dir: data path, string
		is_train: bool, if True, then read train data, otherwise read test data
		batchsize: int, batch size
	Return:
		image_batch: 4D tensor, batchsize * height * width * channels, dtype = tf.float32
		label_batch: 1D tensor, batchsize, dtype = tf.int32
	'''
	img_width = 32
	img_height = 32
	capacity = 2000
	img_depth = 3
	label_bytes = 1
	img_bytes = img_width * img_height * img_depth

	with tf.name_scope('input'):
		if is_train:
			filenames = [os.path.join(file_dir, 'data_batch_%d.bin' % i) for i in np.arange(1,6)]
		else:
			filenames = [os.path.join(file_dir, 'test_batch.bin')]

		filename_queue = tf.train.string_input_producer(filenames)

		reader = tf.FixedLengthRecordReader(label_bytes + img_bytes)

		key, value = reader.read(filename_queue)

		record_bytes = tf.decode_raw(value, tf.uint8)

		label = tf.slice(record_bytes, [0], [label_bytes])
		label = tf.cast(label, tf.int32)

		img_raw = tf.slice(record_bytes, [label_bytes], [img_bytes])
		img_raw = tf.reshape(img_raw, [img_depth, img_height, img_width])
		img = tf.transpose(img_raw, (1,2,0)) #convert from D/H/W to H/W/D
		img = tf.cast(img, tf.float32)

		# data argumentation
#        image = tf.random_crop(image, [24, 24, 3])# randomly crop the image size to 24 x 24
#        image = tf.image.random_flip_left_right(image)
#        image = tf.image.random_brightness(image, max_delta=63)
#        image = tf.image.random_contrast(image,lower=0.2,upper=1.8)

		img = tf.image.per_image_standardization(img)
		if is_shuffle:
			image_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size = batchsize, 
				num_threads = 16, capacity = capacity, min_after_dequeue = 1500, name = 'input')
		else:
			image_batch, label_batch = tf.train.batch([img, label], batch_size = batchsize, 
				num_threads = 16, capacity = capacity, name = 'input')

                return image_batch, tf.reshape(label_batch, [batchsize])
		#One-hot
	        # n_classes = 10
		#label_batch = tf.one_hot(label_batch, depth = n_classes)
                 
		#return image_batch, tf.reshape(label_batch, [batchsize, n_classes])

'''
#Test
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

data_dir = 'cifar-10-batches-bin/'
BATCH_SIZE_TEST = 2
image_batch, label_batch = read_files(data_dir, batchsize = BATCH_SIZE_TEST)
print label_batch
with tf.Session() as sess:
	i = 0;
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)

	try:
		while not coord.should_stop() and i < 1:
			img, label = sess.run([image_batch, label_batch])

			for j in np.arange(BATCH_SIZE_TEST):
				print 'labels:', label[j]
				plt.imshow(img[j, :, :, :])
				plt.show()
			i = i + 1
	except tf.errors.OutOfRangeError:
		print 'done!'
	finally:
		coord.request_stop()
	coord.join(threads)
'''
