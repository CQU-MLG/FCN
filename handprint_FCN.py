from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "hand_print_logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/hand_print/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e4 + 1)
NUM_OF_CLASSESS = 3
IMAGE_SIZE = 800

#定义卷积层函数
def conv_op(input_op,name,kh,kw,n_out,dh,dw,p): #分别表示输入张量，卷积层名字，
#卷积核高和宽，输出卷积核数量，步长以及步长的高和宽，以及参数列表p
    n_in=input_op.get_shape()[-1].value  #获取输入的通道数量，即输入的卷积核的数量或者图像通道
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+"w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv=tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
        bias_init_val=tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases=tf.Variable(bias_init_val,trainable=True,name='b')
        z=tf.nn.bias_add(conv,biases)
        activation=tf.nn.relu(z,name=scope)
        p+=[kernel,biases]
        return activation

#创建最大池化层函数
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)


#根据载入的权重建立原始的 VGGNet 的网络
## vgg 网络部分， weights 是vgg网络各层的权重集合， image是被预测的图像的向量

#def vgg_net(weights, input_op):
# 开始定义VGG-16
def inference_op(input_op):
    net = {}
    p = []
    # 第一段卷积，两个卷积层和一个池化层
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)
    net["pool1"] =pool1
    # 第二段卷积
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dw=2, dh=2)
    net["pool2"] = pool2
    # 第三段卷积
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dw=2, dh=2)
    net["pool3"] = pool3
    # 第四段卷积
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dw=2, dh=2)
    net["pool4"] = pool4
    # 第五段卷积操作
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    return net,conv5_3

'''
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            # reshape(-1)把bias参数数组合并成一行。
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)

        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current'''

        ## vgg 的前5层的stride都是2，也就是前5层的size依次减小1倍
        ## 这里处理了前4层的stride，用的是平均池化
        ## 第5层的pool在下文的外部处理了，用的是最大池化
        ## pool1 size缩小2倍
        ## pool2 size缩小4倍
        ## pool3 size缩小8倍
        ## pool4 size缩小16倍


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    '''
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])
    print("weights.shape",weights.shape)

    processed_image = utils.process_image(image, mean_pixel)'''

    with tf.variable_scope("inference"):
        pooling_net,conv_final_layer = inference_op(image)
        #conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = pooling_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        # 对第8层的结果进行反卷积(上采样),通道数也由NUM_OF_CLASSESS变为第4层的通道数
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(pooling_net["pool4"]))
        fuse_1 = tf.add(conv_t1, pooling_net["pool4"], name="fuse_1")

        deconv_shape2 = pooling_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(pooling_net["pool3"]))
        fuse_2 = tf.add(conv_t2, pooling_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
    print("annotation_pred.shape",annotation_pred.shape)
    print("conv_t3",conv_t3)
    print("tf.expand_dims(annotation_pred, dim=3)",tf.expand_dims(annotation_pred, dim=3))
    return tf.expand_dims(annotation_pred, dim=3), conv_t3

#训练：定义训练损失优化器及训练的梯度下降方法以更新参数  更新参数 使loss最小
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    #pre_ann是一个M*N*1的数组 后者是M*N*151的数组
    pred_annotation, logits = inference(image, keep_probability)
    print("pred_annotation.shape",pred_annotation.shape)
    correct_prediction = tf.equal(tf.argmax(annotation, 1), pred_annotation)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #tf.summary.image("input_image", image, max_outputs=2)
    #tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    #tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    print("pred_annotation",tf.cast(pred_annotation, tf.uint8))
    ## 定义损失函数，这里使用交叉熵的平均值作为损失函数
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                         name="entropy")))
    #绘制损失函数
    loss_summary = tf.summary.scalar("entropy", loss)
    #返回需要训练的变量列表
    trainable_var = tf.trainable_variables()
    '''
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)'''

    #训练操作定义
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    #summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    #读图片
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    '''
      返回一个字典:
      image_list{ 
               "training":[{'image': image_full_name, 'annotation': annotation_file, 'image_filename': },......],
               "validation":[{'image': image_full_name, 'annotation': annotation_file, 'filename': filename},......]
               }
    '''
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        ## 批量读取数据集的类
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation',sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    # logs/  训练和测试  如果存在，则导入  可用于测试和恢复sess  训练断点恢复
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                validation_writer.add_summary(summary_sva, itr)
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                   keep_probability: 1.0})
        #前 (2, 224, 224, 1) (2, 224, 224, 1)
        #后 (2, 224, 224) (2, 224, 224)
        print("前",pred.shape,valid_annotations.shape)
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)
        print("后",pred.shape,valid_annotations.shape)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr),mean=False)
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr),mean=False)
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr),mean=False)

            print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()
