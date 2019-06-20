import tensorflow as tf
import DataGenerator
import time

learning_rate = 1e-5
batch_size = 2
prefetch = 4
no_of_epochs = 10000
smoothing = 0.00001

# placeholder for training mode
is_training = tf.placeholder(tf.bool)

# input data generator
trainTransforms = [
    DataGenerator.RandomFlip(),
    DataGenerator.HistogramMatching(data_dir='train-data', train_size=40, prob=0.5),
    DataGenerator.RandomSmoothing(prob=0.3),
    DataGenerator.RandomNoise(prob=0.5),
    DataGenerator.Normalization()
    ]

valTransforms = [
    DataGenerator.Normalization()
    ]

TrainDataset = DataGenerator.DataGenerator(
    data_dir='train-data',
    transforms=trainTransforms,
    train=True
    )

ValDataset = DataGenerator.DataGenerator(
    data_dir='val-data',
    transforms=valTransforms,
    train=False
    )

trainDataset = TrainDataset.get_dataset()
trainDataset = trainDataset.shuffle(buffer_size=5)
trainDataset = trainDataset.batch(batch_size)
trainDataset = trainDataset.prefetch(prefetch)

valDataset = ValDataset.get_dataset()
valDataset = valDataset.shuffle(buffer_size=5)
valDataset = valDataset.batch(batch_size)
valDataset = valDataset.prefetch(prefetch)

iterator = tf.data.Iterator.from_structure(trainDataset.output_types, trainDataset.output_shapes)

training_init_op = iterator.make_initializer(trainDataset)
validation_init_op = iterator.make_initializer(valDataset)
next_item = iterator.get_next()

# convolution layer
def conv3d(x, no_of_input_channels, no_of_filters, filter_size, strides, padding, name):
    with tf.variable_scope(name) as scope:
        
        initializer = tf.variance_scaling_initializer()
        
        filter_size.extend([no_of_input_channels, no_of_filters])
        weights = tf.Variable(initializer(filter_size), name='weights')
        biases = tf.Variable(initializer([no_of_filters]), name='biases')
        conv = tf.nn.conv3d(x, weights, strides=strides, padding=padding, name=name)
        conv += biases
                
        return conv

# transposed convolution layer
def upsamp(x, no_of_kernels, name):
    with tf.variable_scope(name) as scope:
        upsamp = tf.layers.conv3d_transpose(x, no_of_kernels, [2,2,2], 2, padding='VALID', use_bias=True, reuse=tf.AUTO_REUSE)
        return upsamp

# PReLu layer
def prelu(x, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu", reuse=tf.AUTO_REUSE):
        alpha = tf.get_variable("prelu", shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
        prelu_out = tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)
        return prelu_out
    
# model graph
def graph_encoder(x):
        
    fine_grained_features = {}
    
    conv1 = conv3d(x,1,16,[3,3,3],[1,1,1,1,1],'SAME','Conv1_1')
    conv1 = conv3d(conv1,16,16,[3,3,3],[1,1,1,1,1],'SAME','Conv1_2')
    conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    conv1 = prelu(conv1,'prelu1')
    
    res1 = tf.add(x,conv1)
    fine_grained_features['res1'] = res1
    
    down1 = conv3d(res1,16,32,[2,2,2],[1,2,2,2,1],'VALID','DownSampling1')
    down1 = prelu(down1,'down_prelu1')
    
    conv2 = conv3d(down1,32,32,[3,3,3],[1,1,1,1,1],'SAME','Conv2_1')
    conv2 = conv3d(conv2,32,32,[3,3,3],[1,1,1,1,1],'SAME','Conv2_2')
    conv2= tf.layers.batch_normalization(conv2, training=is_training)
    conv2 = prelu(conv2,'prelu2')
    
    conv3 = conv3d(conv2,32,32,[3,3,3],[1,1,1,1,1],'SAME','Conv3_1')
    conv3 = conv3d(conv3,32,32,[3,3,3],[1,1,1,1,1],'SAME','Conv3_2')
    conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    conv3 = prelu(conv3,'prelu3')
    
    res2 = tf.add(down1,conv3)
    fine_grained_features['res2'] = res2

    down2 = conv3d(res2,32,64,[2,2,2],[1,2,2,2,1],'VALID','DownSampling2')
    down2 = prelu(down2,'down_prelu2')
    
    conv4 = conv3d(down2,64,64,[3,3,3],[1,1,1,1,1],'SAME','Conv4_1')
    conv4 = conv3d(conv4,64,64,[3,3,3],[1,1,1,1,1],'SAME','Conv4_2')
    conv4 = tf.layers.batch_normalization(conv4, training=is_training)
    conv4 = prelu(conv4,'prelu4')
    
    conv5 = conv3d(conv4,64,64,[3,3,3],[1,1,1,1,1],'SAME','Conv5_1')
    conv5 = conv3d(conv5,64,64,[3,3,3],[1,1,1,1,1],'SAME','Conv5_2')
    conv5 = tf.layers.batch_normalization(conv5, training=is_training)
    conv5 = prelu(conv5,'prelu5')
    
    conv6 = conv3d(conv5,64,64,[3,3,3],[1,1,1,1,1],'SAME','Conv6_1')
    conv6 = conv3d(conv6,64,64,[3,3,3],[1,1,1,1,1],'SAME','Conv6_2')
    conv6 = tf.layers.batch_normalization(conv6, training=is_training)
    conv6 = prelu(conv6,'prelu6')
    
    res3 = tf.add(down2,conv6)
    fine_grained_features['res3'] = res3

    down3 = conv3d(res3,64,128,[2,2,2],[1,2,2,2,1],'VALID','DownSampling3')
    down3 = prelu(down3,'down_prelu3')
    
    conv7 = conv3d(down3,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv7_1')
    conv7 = conv3d(conv7,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv7_2')
    conv7 = tf.layers.batch_normalization(conv7, training=is_training)
    conv7 = prelu(conv7,'prelu7')
    
    conv8 = conv3d(conv7,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv8_1')
    conv8 = conv3d(conv8,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv8_2')
    conv8 = tf.layers.batch_normalization(conv8, training=is_training)
    conv8 = prelu(conv8,'prelu8')
    
    conv9 = conv3d(conv8,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv9_1')
    conv9 = conv3d(conv9,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv9_2')
    conv9 = tf.layers.batch_normalization(conv9, training=is_training)
    conv9 = prelu(conv9,'prelu9')
    
    res4 = tf.add(down3,conv9)
    fine_grained_features['res4'] = res4

    down4 = conv3d(res4,128,256,[2,2,2],[1,2,2,2,1],'VALID','DownSampling4')
    down4 = prelu(down4,'down_prelu4')
    
    conv10 = conv3d(down4,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv10_1')
    conv10 = conv3d(conv10,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv10_2')
    conv10 = tf.layers.batch_normalization(conv10, training=is_training)
    conv10 = prelu(conv10,'prelu10')
    
    conv11 = conv3d(conv10,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv11_1')
    conv11 = conv3d(conv11,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv11_2')
    conv11 = tf.layers.batch_normalization(conv11, training=is_training)
    conv11 = prelu(conv11,'prelu11')
    
    conv12 = conv3d(conv11,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv12_1')
    conv12 = conv3d(conv12,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv12_2')
    conv12 = tf.layers.batch_normalization(conv12, training=is_training)
    conv12 = prelu(conv12,'prelu12')
    
    res5 = tf.add(down4,conv12)
    fine_grained_features['res5'] = res5
    
    return fine_grained_features

def graph_decoder(features):
        
    inp = features['res5']
    
    upsamp1 = upsamp(inp,128,'Upsampling1')
    upsamp1 = prelu(upsamp1,'prelu_upsamp1')
    
    concat1 = tf.concat([upsamp1,features['res4']],axis=4)
    
    conv13 = conv3d(concat1,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv13_1')
    conv13 = conv3d(conv13,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv13_2')
    conv13 = tf.layers.batch_normalization(conv13, training=is_training)
    conv13 = prelu(conv13,'prelu13')
    
    conv14 = conv3d(conv13,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv14_1')
    conv14 = conv3d(conv14,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv14_2')
    conv14 = tf.layers.batch_normalization(conv14, training=is_training)
    conv14 = prelu(conv14,'prelu14')
    
    conv15 = conv3d(conv14,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv15_1')
    conv15 = conv3d(conv15,256,256,[3,3,3],[1,1,1,1,1],'SAME','Conv15_2')
    conv15 = tf.layers.batch_normalization(conv15, training=is_training)
    conv15 = prelu(conv15,'prelu15')
    
    res6 = tf.add(concat1,conv15)
    
    upsamp2 = upsamp(res6,64,'Upsampling2')
    upsamp2 = prelu(upsamp2,'prelu_upsamp2')
    
    concat2 = tf.concat([upsamp2,features['res3']],axis=4)
    
    conv16 = conv3d(concat2,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv16_1')
    conv16 = conv3d(conv16,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv16_2')
    conv16 = tf.layers.batch_normalization(conv16, training=is_training)
    conv16 = prelu(conv16,'prelu16')
    
    conv17 = conv3d(conv16,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv17_1')
    conv17 = conv3d(conv17,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv17_2')
    conv17 = tf.layers.batch_normalization(conv17, training=is_training)
    conv17 = prelu(conv17,'prelu17')
    
    conv18 = conv3d(conv17,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv18_1')
    conv18 = conv3d(conv18,128,128,[3,3,3],[1,1,1,1,1],'SAME','Conv18_2')
    conv18 = tf.layers.batch_normalization(conv18, training=is_training)
    conv18 = prelu(conv18,'prelu18')
    
    res7 = tf.add(concat2,conv18)
    
    upsamp3 = upsamp(res7,32,'Upsampling3')
    upsamp3 = prelu(upsamp3,'prelu_upsamp3')
    
    concat3 = tf.concat([upsamp3,features['res2']],axis=4)
    
    conv19 = conv3d(concat3,64,64,[3,3,3],[1,1,1,1,1],'SAME','Conv19_1')
    conv19 = conv3d(conv19,64,64,[3,3,3],[1,1,1,1,1],'SAME','Conv19_2')
    conv19 = tf.layers.batch_normalization(conv19, training=is_training)
    conv19 = prelu(conv19,'prelu19')
    
    conv20 = conv3d(conv19,64,64,[3,3,3],[1,1,1,1,1],'SAME','Conv20_1')
    conv20 = conv3d(conv20,64,64,[3,3,3],[1,1,1,1,1],'SAME','Conv20_2')
    conv20 = tf.layers.batch_normalization(conv20, training=is_training)
    conv20 = prelu(conv20,'prelu20')
    
    res8 = tf.add(concat3,conv20)
    
    upsamp4 = upsamp(res8,16,'Upsampling4')
    upsamp4 = prelu(upsamp4,'prelu_upsamp4')
    
    concat4 = tf.concat([upsamp4,features['res1']],axis=4)
    
    conv21 = conv3d(concat4,32,32,[3,3,3],[1,1,1,1,1],'SAME','Conv21_1')
    conv21 = conv3d(conv21,32,32,[3,3,3],[1,1,1,1,1],'SAME','Conv21_2')
    conv21 = tf.layers.batch_normalization(conv21, training=is_training)
    conv21 = prelu(conv21,'prelu21')
    
    res9 = tf.add(concat4,conv21)
    
    conv22 = conv3d(res9,32,1,[1,1,1],[1,1,1,1,1],'SAME','Conv22')
    conv22 = tf.nn.sigmoid(conv22,'sigmoid')
    
    return conv22

# loss and optimizer
def model_fn():
    
    features, labels = next_item
        
    features = tf.reshape(features, [-1, 128, 128, 64, 1])
    labels = tf.cast(tf.reshape(labels, [-1, 128, 128, 64, 1]), dtype=tf.float32)
    
    # writing summaries to tensorboard
    tf.summary.image('features', features[:, :, :, 32:33, 0], max_outputs=2,collections=['val'])
    tf.summary.image('labels', labels[:, :, :, 32:33, 0], max_outputs=2,collections=['val'])
    
    labels = tf.reshape(labels, [-1,128*128*64])
    
    encoded = graph_encoder(features)
    decoded = graph_decoder(encoded)

    decoded = tf.reshape(decoded, [-1, 128, 128, 64])
    tf.summary.image('segmentation', decoded[:, :, :, 32:33], max_outputs=2, collections=['val'])
    
    output =  tf.reshape(decoded, [-1,128*128*64])
    
    # dice loss
    cost = tf.reduce_mean(tf.divide(smoothing + tf.multiply(2.0, tf.reduce_sum(output * labels, axis=-1)),
                 tf.add(tf.reduce_sum(output, axis=-1), tf.reduce_sum(labels, axis=-1))))
    
    tf.summary.scalar('training_loss', cost)
    tf.summary.scalar('val_loss', cost,collections=['val'])
    
    # for batchnorm
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        opt = tf.train.AdamOptimizer(learning_rate)

        grads = tf.gradients(1-cost, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))

        training_operation = opt.apply_gradients(grads_and_vars=grads)
    
    for grad, var in grads:
        tf.summary.histogram(var.name.replace(':',"_") + '/gradient', grad)
        tf.summary.histogram(var.name.replace(':',"_") , var)
    
    return cost, training_operation

# running the session 
def train():
    with tf.Session() as sess:
        
        cost, opt = model_fn()
        sess.run(tf.global_variables_initializer())

        # merging tensorflow summaries
        merged = tf.summary.merge_all()
        merged_val = tf.summary.merge_all(key = 'val')

        train_writer = tf.summary.FileWriter('event/train',sess.graph)
        val_writer = tf.summary.FileWriter('event/val')
        
        saver = tf.train.Saver()

        for epoch in range(1, no_of_epochs+1):
            start_time = time.time()
            train_loss = []
            examples = 0

            # initializing iterator with training dataset
            sess.run([training_init_op])

            while(True):
                try:
                    # training procedure
                    examples += 1
                    loss, _, summary = sess.run([cost, opt, merged], feed_dict={is_training: True})
                    train_writer.add_summary(summary,epoch)
                    train_loss.append(loss)
                    print('Epoch: {} - ex: {} - loss: {:.6f}'.format(epoch, examples*batch_size, sum(train_loss)/len(train_loss)), end="\r")
                except tf.errors.OutOfRangeError:
                    val_loss = []
                    val_example = 0

                    # initializing iterator with validation dataset
                    sess.run([validation_init_op])

                    while(True):
                        try:
                            val_example += 1
                            loss, summary_l = sess.run([cost, merged_val], feed_dict={is_training: False})
                            val_writer.add_summary(summary_l,epoch)
                            val_loss.append(loss)
                            print('Epoch: {} - ex: {} - val_loss: {:.6f}'.format(epoch, val_example*batch_size, sum(val_loss)/len(val_loss)), end="\r")

                        except tf.errors.OutOfRangeError:
                            break
                    break
            
            print('Epoch: {}/{} - loss: {:.6f} - val_loss: {:.6f} - time: {:.4f}'.format(epoch, no_of_epochs, 
                sum(train_loss)/len(train_loss), sum(val_loss)/len(val_loss), time.time()-start_time))
            
            # saving weights
            if epoch%20==0:
                saver.save(sess, '/temp/weights_epoch_{0}.ckpt'.format(epoch))

if __name__ == '__main__':
    train()
