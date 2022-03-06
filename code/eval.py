# -*- coding: UTF-8 -*-

import os
import sys
import shutil
import random
import scipy
from datetime import datetime
import progressbar
from image_data_handler_joint_multimodal import ImageDataHandler
from resnet18 import ResNet
from layer_blocks import *
from tensorflow.data import Iterator
from utils import flat_shape, count_params
import cPickle as pickle

from imgaug import imgaug as ia
from imgaug import augmenters as iaa

from keras.objectives import categorical_crossentropy
from keras.optimizers import RMSprop
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.metrics import categorical_accuracy
from keras import backend as K

from tensorflow.python.client import device_lib
print device_lib.list_local_devices()



""" Configuration setting """

tf.set_random_seed(7)

# Data-related params
if len(sys.argv) != 3:
    print("The script requires 2 arguments: (1) the dataset root directory and (2) the parameters root directory.")
dataset_root_dir = sys.argv[1] #'/mnt/datasets/ocid_dataset'
params_root_dir = sys.argv[2] #'/mnt/params/models'
# python code/train_and_eval.py <dataset_dir> <params_dir>

#dataset_train_dir_rgb = dataset_root_dir + '/ARID20_crops/squared_rgb/'
dataset_train_dir_rgb = dataset_root_dir + '/train_rgb/'
#dataset_val_dir_rgb = dataset_root_dir + '/ARID10_crops/squared_rgb/'
dataset_val_dir_rgb = dataset_root_dir + '/val_rgb/'
params_dir_rgb = params_root_dir + '/resnet18_ocid_rgb++_params.npy'

#dataset_train_dir_depth = dataset_root_dir + '/ARID20_crops/surfnorm++/'
dataset_train_dir_depth = dataset_root_dir + '/train_hha/'
#dataset_val_dir_depth = dataset_root_dir + '/ARID10_crops/surfnorm++/'
dataset_val_dir_depth = dataset_root_dir + '/val_hha/'
params_dir_depth = params_root_dir + '/resnet18_ocid_surfnorm++_params.npy'

# train_file = dataset_root_dir + '/split_files_and_labels/arid20_clean_sync_instances.txt'
# val_file = dataset_root_dir + '/split_files_and_labels/arid10_clean_sync_instances.txt'
train_file = dataset_root_dir + '/train_labels.txt'
val_file = dataset_root_dir + '/val_labels.txt'

# Log params
tensorboard_log = '/tmp/tensorflow/'

# Solver params
learning_rate = [[0.0001]]
# num_epochs = 50
num_epochs = 2
# batch_size = [[32]]
batch_size = [[2]]
num_neurons = [[100]]
l2_factor = [[0.0]]
maximum_norm = [[4]]
dropout_rate = [[0.4]]

depth_transf = [[256]]
transf_block = transformation_block_v1

# Checkpoint dir
# checkpoint_dir = "/tmp/my_caffenet/"
checkpoint_dir = "./checkpoint/"
if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
## not后面的表达式为False时，执行冒号后面的语句
## isdir判断是否为一个目录；mkdir为创建目录

checkpoint_interval = 1
# 保存间隔

# Input/Output
# num_classes = 49
# img_size = [224, 224]
# num_channels = 3
num_classes = 49
img_size = [224, 224] #不是原图片大小，修改后运算出错
num_channels = 3

##
class_names = ['0', '1/seacu', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
               '20', '21', '22', '23', '24', '25', '26', '27', '28', '29','30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
               '40', '41', '42', '43', '44', '45', '46', '47', '48']

""" Online data augmentation """


def random_choice(start, end, _multiply):
    start = start * _multiply
    end = end * _multiply
    num = random.randrange(start, end + 1, 1)
    # print ("il num e'",num/_multiply)
    return float(num / _multiply)


def x_y_random_image(image):
    width = image.shape[1]
    hight = image.shape[0]

    border_x = int((256 * 5) / 100.0)
    border_y = int((256 * 9) / 100.0)

    pos_x = random.randrange(0 + border_x, width - border_x, 1)
    pos_y = random.randrange(0 + border_y, hight - border_y, 1)
    # print ("la pos e' ", pos_x , pos_y)
    return pos_x, pos_y


def data_aug(batch, batch_depth):
    num_img = batch.shape[0]
    list = []
    list_depth = []
    for i in range(num_img):
        val_fliplr = random.randrange(0, 2, 1)  # in questo modo il due non e compreso e restituisce i valori 0 o 1
        list.extend([iaa.Fliplr(val_fliplr)])
        list_depth.extend([iaa.Fliplr(val_fliplr)])

        val_fliplr = random.randrange(0, 2, 1)  # in questo modo il due non e compreso e restituisce i valori 0 o 1
        list.extend([iaa.Flipud(val_fliplr)])
        list_depth.extend([iaa.Flipud(val_fliplr)])

        val_scala = random.randrange(5, 11, 1)
        val = float(val_scala / 10.0)
        list.extend([iaa.Affine(val, mode='edge')])
        list.extend([iaa.Affine(10.0 / val_scala, mode='edge')])
        list_depth.extend([iaa.Affine(val, mode='edge')])
        list_depth.extend([iaa.Affine(10.0 / val_scala, mode='edge')])

        val_rotation = random.randrange(-180, 181, 90)
        list.extend([iaa.Affine(rotate=val_rotation, mode='edge')])
        list_depth.extend([iaa.Affine(rotate=val_rotation, mode='edge')])

        augseq = iaa.Sequential(list)
        batch[i] = augseq.augment_image(batch[i])
        augseq_depth = iaa.Sequential(list)
        batch_depth[i] = augseq_depth.augment_image(batch_depth[i])

        list = []
        list_depth = []


""" Loop for gridsearch of hyper-parameters """

set_params = [lr + nn + bs + aa + mn + do + dt for lr in learning_rate for nn in num_neurons for bs in batch_size for aa
              in l2_factor for mn in maximum_norm for do in dropout_rate for dt in depth_transf]

for hp in set_params:

    lr = hp[0]
    nn = hp[1]
    bs = hp[2]
    aa = hp[3]
    mn = hp[4]
    do = hp[5]
    dt = hp[6]

    """ Data management """

    # Place data loading and preprocessing on the cpu
    # with tf.device('/cpu:0'):
    dataset_train_dir = [dataset_train_dir_rgb, dataset_train_dir_depth]
    dataset_val_dir = [dataset_val_dir_rgb, dataset_val_dir_depth]

    tr_data = ImageDataHandler(
        train_file,
        data_dir=dataset_train_dir,
        params_dir=params_root_dir,
        img_size=img_size,
        batch_size=bs,
        num_classes=num_classes,
        shuffle=True,
        random_crops=False)

    val_data = ImageDataHandler(
        val_file,
        data_dir=dataset_val_dir,
        params_dir=params_root_dir,
        img_size=img_size,
        batch_size=bs,
        num_classes=num_classes,
        shuffle=False,
        random_crops=False)

    # Create a re-initializable iterator given the dataset structure
    # no need for two different to deal with training and val_rgb data,
    # just two initializers
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator.make_initializer(val_data.data)

    # Get the number of training/validation steps per epoch （每个epoch多少batch）
    tr_batches_per_epoch = int(np.floor(tr_data.data_size / bs))
    val_batches_per_epoch = int(np.floor(val_data.data_size / bs))

    init_op_rgb = {'training': training_init_op, 'validation': validation_init_op}
    batches_per_epoch = {'training': tr_batches_per_epoch, 'validation': val_batches_per_epoch}

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

    # Start Tensorflow session # 创建一个新的TensorFlow会话
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1})) as sess:
    with tf.Session() as sess:
        """ Network definition """

        ## RGB branch

        # TF placeholder for graph input and output
        x_rgb = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
        x_depth = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
        y = tf.placeholder(tf.float32, [None, num_classes])

        x_rgb = tf.reshape(x_rgb, [-1, img_size[0], img_size[1], 3])
        x_depth = tf.reshape(x_depth, [-1, img_size[0], img_size[1], 3])
        y = tf.reshape(y, [-1, num_classes])

        keep_prob = tf.placeholder(tf.float32)
        training_phase = tf.placeholder(tf.bool)

        # Max norm
        ckn = np.inf
        rkc = mn  # np.inf
        rrc = mn  # np.inf
        dkc = mn  # np.inf
        conv_kernel_constraint = tf.keras.constraints.MaxNorm(ckn, axis=[0, 1, 2])
        rnn_kernel_constraint = tf.keras.constraints.MaxNorm(rkc, axis=[0, 1])
        rnn_recurrent_constraint = tf.keras.constraints.MaxNorm(rrc, axis=1)
        dense_kernel_constraint = tf.keras.constraints.MaxNorm(dkc, axis=0)

        # Initialize models
        with tf.variable_scope('rgb', reuse=None):
            model_rgb = ResNet(x_rgb, num_classes, mode='rgb')
        with tf.variable_scope('depth', reuse=None):
            model_depth = ResNet(x_depth, num_classes, mode='depth')

        # Extract features
        res1_rgb = model_rgb.relu1
        inter_res2_rgb = model_rgb.inter_res2
        res2_rgb = model_rgb.res2
        inter_res3_rgb = model_rgb.inter_res3
        res3_rgb = model_rgb.res3
        inter_res4_rgb = model_rgb.inter_res4
        res4_rgb = model_rgb.res4
        inter_res5_rgb = model_rgb.inter_res5
        res5_rgb = model_rgb.res5

        pool2_flat_rgb = model_rgb.pool2_flat

        res1_depth = model_depth.relu1
        inter_res2_depth = model_depth.inter_res2
        res2_depth = model_depth.res2
        inter_res3_depth = model_depth.inter_res3
        res3_depth = model_depth.res3
        inter_res4_depth = model_depth.inter_res4
        res4_depth = model_depth.res4
        inter_res5_depth = model_depth.inter_res5
        res5_depth = model_depth.res5

        pool2_flat_depth = model_depth.pool2_flat

        # Conv1x1
        with tf.variable_scope('conv1x1'):

            # depth_transf = 64

            relu_conv1x1_res1_rgb = transformation_block(res1_rgb, dt, conv_kernel_constraint, training_phase,
                                                         'redux_rgb_res1')
            relu_conv1x1_inter_res2_rgb = transf_block(inter_res2_rgb, dt, conv_kernel_constraint, training_phase,
                                                       'redux_rgb_inter_res2')
            relu_conv1x1_res2_rgb = transf_block(res2_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res2')
            relu_conv1x1_inter_res3_rgb = transf_block(inter_res3_rgb, dt, conv_kernel_constraint, training_phase,
                                                       'redux_rgb_inter_res3')
            relu_conv1x1_res3_rgb = transf_block(res3_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res3')
            relu_conv1x1_inter_res4_rgb = transf_block(inter_res4_rgb, dt, conv_kernel_constraint, training_phase,
                                                       'redux_rgb_inter_res4')
            relu_conv1x1_res4_rgb = transf_block(res4_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res4')
            relu_conv1x1_inter_res5_rgb = transf_block(inter_res5_rgb, dt, conv_kernel_constraint, training_phase,
                                                       'redux_rgb_inter_res5')
            relu_conv1x1_res5_rgb = transf_block(res5_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res5')

            relu_conv1x1_res1_depth = transformation_block(res1_depth, dt, conv_kernel_constraint, training_phase,
                                                           'redux_depth_res1')
            relu_conv1x1_inter_res2_depth = transf_block(inter_res2_depth, dt, conv_kernel_constraint, training_phase,
                                                         'redux_depth_inter_res2')
            relu_conv1x1_res2_depth = transf_block(res2_depth, dt, conv_kernel_constraint, training_phase,
                                                   'redux_depth_res2')
            relu_conv1x1_inter_res3_depth = transf_block(inter_res3_depth, dt, conv_kernel_constraint, training_phase,
                                                         'redux_depth_inter_res3')
            relu_conv1x1_res3_depth = transf_block(res3_depth, dt, conv_kernel_constraint, training_phase,
                                                   'redux_depth_res3')
            relu_conv1x1_inter_res4_depth = transf_block(inter_res4_depth, dt, conv_kernel_constraint, training_phase,
                                                         'redux_depth_inter_res4')
            relu_conv1x1_res4_depth = transf_block(res4_depth, dt, conv_kernel_constraint, training_phase,
                                                   'redux_depth_res4')
            relu_conv1x1_inter_res5_depth = transf_block(inter_res5_depth, dt, conv_kernel_constraint, training_phase,
                                                         'redux_depth_inter_res5')
            relu_conv1x1_res5_depth = transf_block(res5_depth, dt, conv_kernel_constraint, training_phase,
                                                   'redux_depth_res5')

        relu_conv1x1_res1_rgb = tf.reshape(relu_conv1x1_res1_rgb, [-1, flat_shape(relu_conv1x1_res1_rgb)])
        relu_conv1x1_inter_res2_rgb = tf.reshape(relu_conv1x1_inter_res2_rgb,
                                                 [-1, flat_shape(relu_conv1x1_inter_res2_rgb)])
        relu_conv1x1_res2_rgb = tf.reshape(relu_conv1x1_res2_rgb, [-1, flat_shape(relu_conv1x1_res2_rgb)])
        relu_conv1x1_inter_res3_rgb = tf.reshape(relu_conv1x1_inter_res3_rgb,
                                                 [-1, flat_shape(relu_conv1x1_inter_res3_rgb)])
        relu_conv1x1_res3_rgb = tf.reshape(relu_conv1x1_res3_rgb, [-1, flat_shape(relu_conv1x1_res3_rgb)])
        relu_conv1x1_inter_res4_rgb = tf.reshape(relu_conv1x1_inter_res4_rgb,
                                                 [-1, flat_shape(relu_conv1x1_inter_res4_rgb)])
        relu_conv1x1_res4_rgb = tf.reshape(relu_conv1x1_res4_rgb, [-1, flat_shape(relu_conv1x1_res4_rgb)])
        relu_conv1x1_inter_res5_rgb = tf.reshape(relu_conv1x1_inter_res5_rgb,
                                                 [-1, flat_shape(relu_conv1x1_inter_res5_rgb)])
        relu_conv1x1_res5_rgb = tf.reshape(relu_conv1x1_res5_rgb, [-1, flat_shape(relu_conv1x1_res5_rgb)])

        relu_conv1x1_res1_depth = tf.reshape(relu_conv1x1_res1_depth, [-1, flat_shape(relu_conv1x1_res1_depth)])
        relu_conv1x1_inter_res2_depth = tf.reshape(relu_conv1x1_inter_res2_depth,
                                                   [-1, flat_shape(relu_conv1x1_inter_res2_depth)])
        relu_conv1x1_res2_depth = tf.reshape(relu_conv1x1_res2_depth, [-1, flat_shape(relu_conv1x1_res2_depth)])
        relu_conv1x1_inter_res3_depth = tf.reshape(relu_conv1x1_inter_res3_depth,
                                                   [-1, flat_shape(relu_conv1x1_inter_res3_depth)])
        relu_conv1x1_res3_depth = tf.reshape(relu_conv1x1_res3_depth, [-1, flat_shape(relu_conv1x1_res3_depth)])
        relu_conv1x1_inter_res4_depth = tf.reshape(relu_conv1x1_inter_res4_depth,
                                                   [-1, flat_shape(relu_conv1x1_inter_res4_depth)])
        relu_conv1x1_res4_depth = tf.reshape(relu_conv1x1_res4_depth, [-1, flat_shape(relu_conv1x1_res4_depth)])
        relu_conv1x1_inter_res5_depth = tf.reshape(relu_conv1x1_inter_res5_depth,
                                                   [-1, flat_shape(relu_conv1x1_inter_res5_depth)])
        relu_conv1x1_res5_depth = tf.reshape(relu_conv1x1_res5_depth, [-1, flat_shape(relu_conv1x1_res5_depth)])

        # RGB and depth pipelines' merge point
        relu_conv1x1_res1 = tf.concat([relu_conv1x1_res1_rgb, relu_conv1x1_res1_depth], axis=1)
        relu_conv1x1_inter_res2 = tf.concat([relu_conv1x1_inter_res2_rgb, relu_conv1x1_inter_res2_depth], axis=1)
        relu_conv1x1_res2 = tf.concat([relu_conv1x1_res2_rgb, relu_conv1x1_res2_depth], axis=1)
        relu_conv1x1_inter_res3 = tf.concat([relu_conv1x1_inter_res3_rgb, relu_conv1x1_inter_res3_depth], axis=1)
        relu_conv1x1_res3 = tf.concat([relu_conv1x1_res3_rgb, relu_conv1x1_res3_depth], axis=1)
        relu_conv1x1_inter_res4 = tf.concat([relu_conv1x1_inter_res4_rgb, relu_conv1x1_inter_res4_depth], axis=1)
        relu_conv1x1_res4 = tf.concat([relu_conv1x1_res4_rgb, relu_conv1x1_res4_depth], axis=1)
        relu_conv1x1_inter_res5 = tf.concat([relu_conv1x1_inter_res5_rgb, relu_conv1x1_inter_res5_depth], axis=1)
        relu_conv1x1_res5 = tf.concat([relu_conv1x1_res5_rgb, relu_conv1x1_res5_depth], axis=1)

        rnn_input = tf.stack(
            [relu_conv1x1_res1, relu_conv1x1_inter_res2, relu_conv1x1_res2, relu_conv1x1_inter_res3, relu_conv1x1_res3,
             relu_conv1x1_inter_res4, relu_conv1x1_res4, relu_conv1x1_inter_res5, relu_conv1x1_res5], axis=1)

        # Recurrent net
        with tf.variable_scope("rnn"):
            rnn_h = GRU(nn, activation='tanh', dropout=do, recurrent_dropout=do, name="rnn_h",
                        kernel_constraint=rnn_kernel_constraint, recurrent_constraint=rnn_recurrent_constraint)(
                rnn_input)
            preds = Dense(num_classes, activation='softmax', kernel_constraint=dense_kernel_constraint)(rnn_h)

        # Include keras-related metadata in the session
        K.set_session(sess)

        # Define trainable variables
        trainable_variables_rnn = [v for v in tf.trainable_variables(scope="rnn")]
        trainable_variables_conv1x1 = [v for v in tf.trainable_variables(scope="conv1x1")]
        # trainable_variables_rgb = [v for v in tf.trainable_variables(scope="rgb")]
        # trainable_variables_depth = [v for v in tf.trainable_variables(scope="depth")]

        # Define the training and validation opsi
        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(global_step, global_step + 1)
        lr_boundaries = [int(num_epochs * tr_batches_per_epoch * 0.5)]  # , int(num_epochs*tr_batches_per_epoch*0.6)]
        lr_values = [lr, lr / 10]
        decayed_lr = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)

        # L2-regularization
        alpha_rnn = aa
        alpha_conv1x1 = aa
        l2_rnn = tf.add_n([tf.nn.l2_loss(tv_rnn) for tv_rnn in trainable_variables_rnn
                           if 'bias' not in tv_rnn.name]) * alpha_rnn
        l2_conv1x1 = tf.add_n([tf.nn.l2_loss(tv_conv1x1) for tv_conv1x1 in trainable_variables_conv1x1
                               if 'bias' not in tv_conv1x1.name]) * alpha_conv1x1

        # F2-norm
        # rgb_nodes = [relu_conv1x1_inter_res2_rgb, relu_conv1x1_res2_rgb, relu_conv1x1_inter_res3_rgb, relu_conv1x1_res3_rgb, relu_conv1x1_inter_res4_rgb, relu_conv1x1_res4_rgb, relu_conv1x1_inter_res5_rgb, relu_conv1x1_res5_rgb]
        rgb_nodes = [relu_conv1x1_res1_rgb, relu_conv1x1_inter_res2_rgb, relu_conv1x1_res2_rgb,
                     relu_conv1x1_inter_res3_rgb, relu_conv1x1_res3_rgb, relu_conv1x1_inter_res4_rgb,
                     relu_conv1x1_res4_rgb, relu_conv1x1_inter_res5_rgb, relu_conv1x1_res5_rgb]
        # depth_nodes = [relu_conv1x1_inter_res2_depth, relu_conv1x1_res2_depth, relu_conv1x1_inter_res3_depth, relu_conv1x1_res3_depth, relu_conv1x1_inter_res4_depth, relu_conv1x1_res4_depth, relu_conv1x1_inter_res5_depth, relu_conv1x1_res5_depth]
        depth_nodes = [relu_conv1x1_res1_depth, relu_conv1x1_inter_res2_depth, relu_conv1x1_res2_depth,
                       relu_conv1x1_inter_res3_depth, relu_conv1x1_res3_depth, relu_conv1x1_inter_res4_depth,
                       relu_conv1x1_res4_depth, relu_conv1x1_inter_res5_depth, relu_conv1x1_res5_depth]
        reg = tf.Variable(0.0, trainable=False)

        # sigm_reg = 0.0001*tf.sigmoid(((12*tf.cast(global_step,tf.float32))/(num_epochs*tr_batches_per_epoch))-6, name='reg_sigmoid')
        sigm_reg = 1.0
        reg_values = [sigm_reg, 0.0]
        decayed_reg = tf.train.piecewise_constant(global_step, lr_boundaries, reg_values)
        # recompute_reg = tf.assign(reg, sigm_reg)

        reg_lambda = [0.0 * decayed_reg, 0.0 * decayed_reg, 1e-6 * decayed_reg, 1e-5 * decayed_reg, 1e-5 * decayed_reg,
                      1e-4 * decayed_reg, 1e-4 * decayed_reg, 0.0 * decayed_reg, 0.0 * decayed_reg]

        lr_mult_conv1x1 = decayed_reg

        # Loss
        loss_l2 = l2_rnn + l2_conv1x1
        loss_cls = tf.reduce_mean(categorical_crossentropy(y, preds))

        loss = loss_cls  # + loss_l2
        train_step_rnn = tf.keras.optimizers.RMSprop(lr=decayed_lr).get_updates(loss=loss,
                                                                                params=trainable_variables_rnn)
        train_step_conv1x1 = tf.keras.optimizers.RMSprop(lr=decayed_lr * lr_mult_conv1x1).get_updates(loss=loss,
                                                                                                      params=trainable_variables_conv1x1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.group(train_step_rnn, train_step_conv1x1, increment_global_step)
            # train_step = tf.group(train_step_rnn, train_step_conv1x1, train_step_rgb, train_step_depth, increment_global_step, recompute_reg)
        accuracy = tf.reduce_mean(categorical_accuracy(y, preds))

        # Create summaries for Tensorboard
        tf.summary.scalar("loss_cls", loss_cls)
        tf.summary.scalar("loss_l2", loss_l2)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("learning rate", decayed_lr)
        tf.summary.scalar("lambda", decayed_reg)
        summary_op = tf.summary.merge_all()

        name = str(lr) + '_' + str(bs) + '_' + str(nn)
        train_writer = tf.summary.FileWriter(tensorboard_log + name + '/train_rgb/', graph=sess.graph)
        val_writer = tf.summary.FileWriter(tensorboard_log + name + '/val_rgb/')

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load the pretrained weights into the non-trainable layer
        model_rgb.load_params(sess, params_dir_rgb, trainable=False)
        model_depth.load_params(sess, params_dir_depth, trainable=False)

        # print(
        #     "\nHyper-parameters: lr={}, #neurons={}, bs={}, l2={}, max_norm={}, dropout_rate={}".format(lr, nn, bs, aa,
        #                                                                                                 mn, do))
        # print("Number of trainable parameters = {}".format(
        #     count_params(trainable_variables_rnn) + count_params(trainable_variables_conv1x1)))
        #
        # print("\n{} Generate features from training set".format(datetime.now()))

        tb_train_count = 0
        tb_val_count = 0

        # Loop over number of epochs
        num_samples = 0
        # Training set

        #sess.run(training_init_op)
        ### 不能不要上一行的初始化
        saver = tf.train.Saver()
        # 载入保存的模型
        if os.path.exists('tmp/checkpoint'):
            saver.restore(sess,'tmp/model.ckpt')
        #else:
            #sess.run(training_init_op)

        # # Progress bar setting
        # bar = progressbar.ProgressBar(maxval=tr_batches_per_epoch,
        #                               widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        # bar.start()
        # train_loss = 0
        # for i in range(tr_batches_per_epoch):
        #     bar.update(i + 1)
        #     tb_train_count += 1
        #     rgb_batch, depth_batch, label_batch = sess.run(next_batch)
        #
        #     num_samples += np.shape(rgb_batch)[0]
        #     feed_dict = {x_rgb: rgb_batch, x_depth: depth_batch, y: label_batch, keep_prob: (1 - do),
        #                  training_phase: True, K.learning_phase(): 1}
        #     batch_loss, summary = sess.run([loss, summary_op], feed_dict=feed_dict)
        #     train_loss += batch_loss
        #     train_writer.add_summary(summary, tb_train_count)
        #
        # bar.finish()
        #
        # train_loss /= tr_batches_per_epoch  # num_samples
        # print("training loss = {}\n".format(train_loss))
        #
        val_acc = 0
        val_loss = 0
        num_samples = 0

        sess.run(validation_init_op)

        for i in range(val_batches_per_epoch):
            tb_val_count += 1
            rgb_batch, depth_batch, label_batch = sess.run(next_batch)
            num_samples += np.shape(rgb_batch)[0]

            ##
            print(np.shape(rgb_batch))  # （2，224，224，3）

            feed_dict = {x_rgb: rgb_batch, x_depth: depth_batch, y: label_batch, keep_prob: 1.0, training_phase: False,
                         K.learning_phase(): 0}
            # 加上batch_preds
            batch_loss, batch_acc, summary, batch_preds = sess.run([loss, accuracy, summary_op, preds], feed_dict=feed_dict)

            ##
            # print(len(batch_preds)) # 2
            # 本来是ndarray，转换为tensor
            # batch_preds = tf.convert_to_tensor(batch_preds,tf.float32)

            #print(batch_preds) # 输出batch中各个图的所有分类概率？
            # [0.01965416 0.05609968 0.02274507 0.02169436 0.01934089 0.01510134
            #   0.01998439 0.01615115 0.01850731 0.01812542 0.01640475 0.02474153
            #   0.01792471 0.01755322 0.02958737 0.01706557 0.0134481  0.025263
            #   0.01595823 0.01707936 0.0219399  0.02069154 0.02293621 0.02140451
            #   0.02249086 0.01835708 0.02074925 0.01914156 0.01833457 0.02087736
            #   0.02423697 0.02556087 0.02091018 0.0184774  0.0147729  0.01628143
            #   0.01880638 0.01793713 0.01923336 0.01759283 0.01802209 0.02485891
            #   0.01731538 0.02391252 0.02692604 0.01842901 0.01709574 0.01675142
            #   0.01352696]

            ## axis默认0，列最大值索引。axis=1行最大值索引。
            # print(tf.argmax(batch_preds, axis=1).eval())
            print(tf.argmax(batch_preds, axis=1).eval()) # 输出[1 1]，两行的预测类别都是1




            val_loss += batch_loss
            val_acc += batch_acc
            val_writer.add_summary(summary, tb_val_count * (tr_batches_per_epoch / val_batches_per_epoch))

        val_loss /= val_batches_per_epoch  # num_samples
        val_acc /= val_batches_per_epoch  # num_samples

        print("\n{} Validation loss : {}, Validation Accuracy = {:.4f}".format(datetime.now(), val_loss, val_acc))



    tf.reset_default_graph()
    shutil.rmtree(tensorboard_log)

