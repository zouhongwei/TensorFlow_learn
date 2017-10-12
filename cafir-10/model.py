import tensorflow as tf

def inference(images, batch_size = 16):
    '''build the model
    Args:
	images: inputdata (images), 4D tensor, batch_size * height * width * depth
    Notes:
        In each conv layer, the kernel size is:
        [kernel_size, kernel_size, number of input channels, number of output channels].
        number of input channels are from previuous layer, if previous layer is THE input
        layer, number of input channels should be image's channels.
    Return:
    	softmax_linear
    '''
    #conv1
    with tf.variable_scope('conv1') as scope:
	weights = tf.get_variable('weights',
		shape = [3, 3, 3, 96],
		dtype = tf.float32,
		initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01, dtype = tf.float32))
	biases = tf.get_variable('biases',
		shape = [96],
		dtype = tf.float32,
		initializer = tf.constant_initializer(0.0))
	conv = tf.nn.conv2d(images, weights, strides = [1,1,1,1], padding = 'SAME')
	pre_activation = tf.nn.bias_add(conv, biases)
	conv1 = tf.nn.relu(pre_activation, name = scope.name)

    # pool1 & norm1
    with tf.variable_scope('pooling1_lrn') as scope:
 	pool1 = tf.nn.max_pool(conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2 ,1], padding = 'SAME', name = 'pooling1')
	norm1 = tf.nn.lrn(pool1, depth_radius = 4, bias = 1.0, alpha = 0.001/9.0, beta = 0.75, name = 'norm1')

    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,96, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    
    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')
    
    
    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    
    
    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[384,192],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[192],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
     
        
    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[192, 10],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[10],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    
    return softmax_linear

def losses(logits, label):
    '''compute loss
    Args:
	logits: predictions
	lable: ground truth
    Return:
	loss
    '''
    with tf.variable_scope('loss') as scope:
        
        labels = tf.cast(label, tf.int64)
        
        # to use this loss fuction, one-hot encoding is needed!
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
        #                (logits=logits, labels=labels, name='xentropy_per_example')
                        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
                        
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
        
    return loss

def trainning(loss, lr): 
    '''Training ops, the Op returned by this function is what must be passed to 
       'sess.run()' call to cause the model to train.
        
    Args:
        loss: loss tensor, from losses()
        
    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
    	optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    	global_step = tf.Variable(0, name = 'global_step', trainable = False)
    	train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op
    
