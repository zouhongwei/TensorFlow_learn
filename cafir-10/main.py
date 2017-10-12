import os
import numpy as np
import tensorflow as tf
import input_data
import model
import math

def running_train(data_dir, log_dir, max_step = 10000, batch_size = 16, lr = 0.05):
    '''
    '''
    image_batch, label_batch = input_data.read_files(data_dir, batchsize = batch_size)
    print label_batch
    logits = model.inference(image_batch, batch_size = batch_size)
    loss = model.losses(logits, label_batch)
    train_op = model.trainning(loss, lr = lr)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(log_dir,sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
    	for step in np.arange(max_step):
            if coord.should_stop():
    		break
    	    _, tra_loss = sess.run([train_op, loss])

    	    if step % 50 == 0:
                print('Step %d, train loss = %.2f' %(step, tra_loss))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            
            if step % 2000 == 0 or (step + 1) == max_step:
                checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()	

def eveluation(batch_size = 16):
    with tf.Graph().as_default():
	log_dir = 'logs/'
	test_dir = './cifar-10-batches-bin/'
	n_test = 10000

	image_batch, label_batch = input_data.read_files(test_dir, is_train = False , batchsize = batch_size, is_shuffle = False)
	logits = model.inference(image_batch, batch_size = batch_size)
        #label_batch = tf.reshape(label_batch,[batch_size,10])
        #label_batch = tf.cast(label_batch, tf.int32)
	print label_batch
        top_k_op = tf.nn.in_top_k(logits, label_batch,1)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
              	print 'Reading check points.....'
        	ckpt = tf.train.get_checkpoint_state(log_dir)
        	if ckpt and ckpt.model_checkpoint_path:
        		global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        		saver.restore(sess, ckpt.model_checkpoint_path)
        		print 'loading success, global step is %s' % global_step
        	else:
        		print 'no checkpoint file found!'
        		return

        	coord = tf.train.Coordinator()
        	threads = tf.train.start_queue_runners(sess = sess, coord = coord)

        	try:
        		num_iter = int(math.ceil(n_test / batch_size))
        		true_count = 0
        		total_sample_count = num_iter * batch_size
        		step = 0

        		while step < num_iter and not coord.should_stop():
        			predictions = sess.run([top_k_op])
        			true_count += np.sum(predictions)
                                print true_count
        			step += 1
                                #print 'step', step
        		precision = true_count/ (total_sample_count*1.0)
        		print 'precision = %.4f' % precision
        	except Exception as e:
        		coord.request_stop(e)
        	finally:
        		coord.request_stop()

        	coord.join(threads)
        	sess.close()





def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    maxstep = 25000
    batchsize = 64
    ''' 
    data_dir = 'cifar-10-batches-bin/'
    log_dir = 'logs/'
    print 'Begin training...'
    running_train(data_dir, log_dir, max_step = maxstep, batch_size = batchsize)
    print 'Finished training!....'
    '''
    print 'begin eveluation....'
    eveluation(batch_size = batchsize)
    print 'finished....'
     

if __name__ == '__main__':
    main()
