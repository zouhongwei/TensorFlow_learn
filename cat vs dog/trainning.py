import tensorflow as tf
import os
import numpy as np
import input_data
import model
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 32
CAPACITY = 4000
MAX_STEP = 15000
lr = 0.0001	#learning rate

def run_training():
    print 'let us begin....'
    train_dir = '../data/train/'
    logs_train_dir = './train/'

    train, train_label = input_data.get_files(train_dir)
    train_batch, train_label_batch = input_data.get_batch(train,
							  train_label,
               						  IMG_H,
							  IMG_W,
							  BATCH_SIZE,
							  CAPACITY)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss,lr)
    train_acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all() #????
    sess = tf.Session()

    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
    	for step in np.arange(MAX_STEP):
 	    if coord.should_stop():
                print 'coord stop!'
   		break
    	    _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

    	    if step % 50 == 0:
    	 	print 'Step: %d, train_loss = %.2f, train_accuracy = %.2f\n' % (step, tra_loss, tra_acc*100.0)
    		summary_str = sess.run(summary_op)
    		train_writer.add_summary(summary_str, step)

    	    if step % 2500 == 0 or (step + 1) == MAX_STEP:
    		checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
    		saver.save(sess, checkpoint_path, global_step = step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()


#%% Evaluate one image
# when training, comment the following codes.


#from PIL import Image
#import matplotlib.pyplot as plt
#
#def get_one_image(train):
#    '''Randomly pick one image from training data
#    Return: ndarray
#    '''
#    n = len(train)
#    ind = np.random.randint(0, n)
#    img_dir = train[ind]
#
#    image = Image.open(img_dir)
#    plt.imshow(image)
#    image = image.resize([208, 208])
#    image = np.array(image)
#    return image
#
#def evaluate_one_image():
#    '''Test one image against the saved models and parameters
#    '''
#    
#    # you need to change the directories to yours.
#    train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
#    train, train_label = input_data.get_files(train_dir)
#    image_array = get_one_image(train)
#    
#    with tf.Graph().as_default():
#        BATCH_SIZE = 1
#        N_CLASSES = 2
#        
#        image = tf.cast(image_array, tf.float32)
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image, [1, 208, 208, 3])
#        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#        
#        logit = tf.nn.softmax(logit)
#        
#        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
#        
#        # you need to change the directories to yours.
#        logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/' 
#                       
#        saver = tf.train.Saver()
#        
#        with tf.Session() as sess:
#            
#            print("Reading checkpoints...")
#            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#            if ckpt and ckpt.model_checkpoint_path:
#                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                saver.restore(sess, ckpt.model_checkpoint_path)
#                print('Loading success, global_step is %s' % global_step)
#            else:
#                print('No checkpoint file found')
#            
#            prediction = sess.run(logit, feed_dict={x: image_array})
#            max_index = np.argmax(prediction)
#            if max_index==0:
#                print('This is a cat with possibility %.6f' %prediction[:, 0])
#            else:
#                print('This is a dog with possibility %.6f' %prediction[:, 1])


#%%
def main():
    run_training()


if __name__ == "__main__":
    print 'begin training....'
    sys.exit(main())	
