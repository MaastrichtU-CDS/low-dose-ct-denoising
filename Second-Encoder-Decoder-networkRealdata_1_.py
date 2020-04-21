# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:57:59 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:20:49 2019

@author: chenj
"""

# 导入相关模块
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
#matplotlib inline
 

VGG_MEAN = [103.939/256, 116.779/256, 123.68/256]

# 导入数据集
def pares_tf(example_proto):
    #定义解析的字典
    dics = {
            'img_noised': tf.FixedLenFeature([], tf.string),
            'img_standard': tf.FixedLenFeature([], tf.string)
            }
    #调用接口解析一行样本
    parsed_example = tf.parse_single_example(serialized=example_proto,features=dics)
    image_noised = tf.decode_raw(parsed_example['img_noised'],out_type=tf.uint8)
    image_noised= tf.reshape(image_noised,shape=[512,512])
    #这里对图像数据做归一化，是关键，没有这句话，精度不收敛，为0.1左右，
    # 有了这里的归一化处理，精度与原始数据一致
    image_noised = tf.cast(image_noised, tf.float32) * (1. / 255) #在流中抛出img张量
    image_standard = tf.decode_raw(parsed_example['img_standard'],out_type=tf.uint8)
    image_standard= tf.reshape(image_standard,shape=[512,512])
    #这里对图像数据做归一化，是关键，没有这句话，精度不收敛，为0.1左右，
    # 有了这里的归一化处理，精度与原始数据一致
    image_standard = tf.cast(image_standard, tf.float32) * (1. / 255) #在流中抛出img张量
    return image_noised,image_standard
dataset=tf.data.TFRecordDataset(filenames=['RealData_High_train2.tfrecords'])
dataset=dataset.map(pares_tf)
dataset=dataset.shuffle(300).repeat(46).batch(4)

iterator = dataset.make_one_shot_iterator()
next_patch = iterator.get_next()

dataset2=tf.data.TFRecordDataset(filenames=['RealData_High_test2.tfrecords'])
dataset2=dataset2.map(pares_tf)
dataset2=dataset2.shuffle(30).repeat(1).batch(4)

iterator2 = dataset2.make_one_shot_iterator()
next_patch2 = iterator2.get_next()


dataset4=tf.data.TFRecordDataset(filenames=['RealData_High_test2.tfrecords'])
dataset4=dataset4.map(pares_tf)
dataset4=dataset4.repeat(1).batch(4)

iterator4 = dataset4.make_one_shot_iterator()
next_patch4 = iterator4.get_next()
 
inputs_ = tf.placeholder(tf.float32, [None, 512, 512, 1])
targets_ = tf.placeholder(tf.float32, [None, 512, 512, 1])
 
def lrelu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


################VGG-16 based content loss#####################################################
class VGGNet:
    """
    构建VGG16的网络结构 并从预训练好的模型提取参数 加载
    """
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def get_conv_kernel(self, name):
        # 卷积核的参数：w 0  b 1
        return tf.constant(self.data_dict[name][0], name='conv')

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name='fc')

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='bias')

    def conv_layer(self, inputs, name):
        """
        构建一个卷积计算层
        :param inputs: 输入的feature_map
        :param name: 卷积层的名字 也是获得参数的key 不能出错
        :return: 
        """
        with tf.name_scope(name):
            """
            多使用name_scope的好处：1、防止参数命名冲突 2、tensorboard可视化时很规整
            如果scope里面有变量需要训练时则用tf.variable_scope
            """
            conv_w = self.get_conv_kernel(name)
            conv_b = self.get_bias(name)
            # tf.layers.conv2d() 这是一个封装更高级的api
            # 里面并没有提供接口来输入卷积核参数 这里不能用 平时训练cnn网络时非常好用
            result = tf.nn.conv2d(input=inputs, filter=conv_w, strides=[1, 1, 1, 1], padding='SAME', name=name)
            result = tf.nn.bias_add(result, conv_b)
            result = tf.nn.relu(result)
            return result

    def pooling_layer(self, inputs, name):
        # tf.layers.max_pooling2d()
        # tf.nn.max_pool 这里的池化层没有参数 两套api都可以用
        return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name=name)

    def fc_layer(self, inputs, name, activation=tf.nn.relu):
        """
        构建全连接层
        :param inputs: 输入 
        :param name: 
        :param activation: 是否有激活函数的封装 
        :return: 
        """
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            # fc: wx+b 线性变换
            result = tf.nn.bias_add(tf.matmul(inputs, fc_w), fc_b)
            if activation is None:
                # vgg16的最后是不需relu激活的
                return result
            else:
                return activation(result)

    def flatten_op(self, inputs, name):
        # 展平操作 为了后续的fc层必须将维度展平
        with tf.name_scope(name):
            # [NHWC]---> [N, H*W*C]

            x_shape = inputs.get_shape().as_list()
            dim = 1
            for d in x_shape[1:]:
                dim *= d
            inputs = tf.reshape(inputs, shape=[-1, dim])
            # 直接用现成api也是可以的
            # return tf.layers.flatten(inputs)
            return inputs

    def build(self, input_rgb):
        """
        构建vgg16网络结构 抽取特征 FP过程
        :param input_rgb: [1, 224, 224, 3] 
        :return: 
        """
#        start_time = time.time()
        #tf.logging.info('building start...')

        # 在通道维度上分离 深度可分离卷积中也需要用到这个api
        r, g, b = tf.split(input_rgb, num_or_size_splits=3, axis=3)
        # 在通道维度上拼接
        # 输入vgg网络的图像是bgr的（与OpenCV一样 倒序的）而不是rgb
        x_bgr = tf.concat(values=[
            b - VGG_MEAN[0],
            g - VGG_MEAN[1],
            r - VGG_MEAN[2],
        ], axis=3)

        assert x_bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # 构建网络
        # stage 1
        self.conv1_1 = self.conv_layer(x_bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')

        # stage 2
        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')

        # stage 3
        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')

        # stage 4
        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')

        # stage 5
        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')

        # flatten_op
        self.flatten = self.flatten_op(self.pool5, 'flatten_op')

        # fc
        self.fc6 = self.fc_layer(self.flatten, 'fc6')
        self.fc7 = self.fc_layer(self.fc6, 'fc7')
        self.fc8 = self.fc_layer(self.fc7, 'fc8', activation=None)
        self.logits = tf.nn.softmax(self.fc8, name='logits')
#        logging.info('building end... 耗时%3d秒' % (time.time() - start_time))


vgg16_npy_path = 'vgg16.npy'
vgg16_data = np.load(vgg16_npy_path,allow_pickle=True, encoding='latin1')

#print(type(vgg16_data))
data_dict = vgg16_data.item()

vgg16_for_noised = VGGNet(data_dict)
image_noised = tf.placeholder(dtype=tf.float32, shape=[4, 224, 224, 3], name='image_noised')
vgg16_for_noised.build(image_noised)

vgg16_for_standard=VGGNet(data_dict)
image_standard = tf.placeholder(dtype=tf.float32, shape=[4, 224, 224, 3], name='image_standard')
vgg16_for_standard.build(image_standard)

noised_contents = [
    # vgg16_for_content_img.conv1_1,
    vgg16_for_noised.conv2_1,
    # vgg16_for_content_img.conv3_1,
    # vgg16_for_content_img.conv3_2,
    # vgg16_for_content_img.conv5_1,
    # vgg16_for_content_img.conv5_3,
]

standard_content= [
    # vgg16_for_content_img.conv1_1,
    vgg16_for_standard.conv2_1,
    # vgg16_for_content_img.conv3_1,
    # vgg16_for_content_img.conv3_2,
    # vgg16_for_content_img.conv5_1,
    # vgg16_for_content_img.conv5_3,
]

content_loss = tf.zeros(shape=1, dtype=tf.float32)

for c, c_result in zip(noised_contents, standard_content):
    # c c_result [NHWC]
    content_loss += tf.reduce_mean(tf.square(c - c_result), axis=[1, 2, 3])
##########################################end of VGGcontentloss##################### 
    
    
####################################build encoder-decoder######################################
### Encoder
with tf.name_scope('en-convolutions'):
    
  
    conv1 =tf.layers.conv2d(inputs_, filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='SAME',
                            use_bias=True,
                            activation=lrelu,)
# now 512x512x32
with tf.name_scope('en_pooling'):
    maxpool1 = tf.layers.max_pooling2d(conv1,
                                      pool_size=(2, 2),
                                      strides=(2,2),)
#now 256*256*32
with tf.name_scope('en-convolutions'):
    
  
    conv2 =tf.layers.conv2d(maxpool1, filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='SAME',
                            use_bias=True,
                            activation=lrelu,)
# now 256x256x32
with tf.name_scope('en_pooling'):
    maxpool2 = tf.layers.max_pooling2d(conv2,
                                      pool_size=(2, 2),
                                      strides=(2,2),)
#now 128x128x32
with tf.name_scope('en-convolutions'):     
    conv3 = tf.layers.conv2d(maxpool2, filters=32,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='SAME',
                            use_bias=True,
                            activation=lrelu,)
# now 128x128x32
with tf.name_scope('en-pooling'):
    maxpool3 = tf.layers.max_pooling2d(conv3,
                                      pool_size=(2, 2),
                                      strides=(2,2),)
# now 64x64x32
with tf.name_scope('en-convolutions'):
    conv4 = tf.layers.conv2d(maxpool3,
                               filters=32,
                               kernel_size=(3, 3),
                               strides=(1,1),
                               padding='SAME',
                               use_bias=True,
                               activation=lrelu,)
 
#  now 64x64x32
with tf.name_scope('encoding'):
    encoded = tf.layers.max_pooling2d(conv4,
                                      pool_size=(2,2),
                                      strides=(2,2),)
# now 32x32x32
 
### Decoder
with tf.name_scope('decoder'):
    conv5 = tf.layers.conv2d(encoded,
                            filters=32,
                            kernel_size=(3, 3),
                            strides=(1,1),
                            padding='SAME',
                            use_bias=True,
                            activation=lrelu)
#  32x32x32
    upsamples1 = tf.layers.conv2d_transpose(conv5,
                                           filters=32,
                                           kernel_size=3,
                                           padding='SAME',
                                           strides=2,
                                           name='upsample1')
# now 64x64x32
    upsamples2 = tf.layers.conv2d_transpose(0.5*upsamples1+0.5*conv4,
                                           filters=32,
                                           kernel_size=3,
                                           padding='SAME',
                                           strides=2,
                                           name='upsamples2')
# now 128x128x32
    upsamples3=tf.layers.conv2d_transpose(0.5*upsamples2+0.5*conv3,
                                           filters=32,
                                           kernel_size=3,
                                           padding='SAME',
                                           strides=2,
                                           name='upsamples3')
# now 256x256x32
    upsamples4=tf.layers.conv2d_transpose(0.5*upsamples3+0.5*conv2,
                                           filters=32,
                                           kernel_size=3,
                                           padding='SAME',
                                           strides=2,
                                           name='upsamples4')
    
    logits = tf.layers.conv2d(0.5*upsamples4+0.5*conv1, 
                             filters=1,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             name='logits',
                             padding='SAME',
                             use_bias=True)
# now 512x512x1
    # 通过sigmoid传递logits以获得重建图像
    decoded = tf.sigmoid(logits, name='recon')
 
# 定义损失函数和优化器
loss = tf.nn.sigmoid_cross_entropy_with_logits(
logits=logits, labels=targets_)
 
learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
 
################end-of-encoder-and-decoder####################################################

   
 
# 训练
sess = tf.Session()
 
saver = tf.train.Saver()
loss = []
valid_loss = []
 
display_step = 1
epochs = 45
batch_size = 4
lr =1e-5
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./graphs', sess.graph)
fo1=open("Real_train_High_Second_3_shortcuts.txt", "w")
for e in range(epochs):
#    if epochs>20:
#        batch_size=4
#    else:
#        batch_size=12
    total_batch = int(5040/batch_size)
    for ibatch in range(total_batch):
        x_train_img,batch_x_img=sess.run(next_patch)
        x_train_noisy=x_train_img.reshape((-1,512,512,1))
        x_test_noisy=x_train_noisy
        imgs = batch_x_img.reshape((-1, 512, 512, 1))
        imgs_test=imgs
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: x_train_noisy,
                                                         targets_: imgs,learning_rate:lr})
      
        batch_cost_test = sess.run(cost, feed_dict={inputs_: x_test_noisy,
                                                         targets_: imgs_test})
    fo1.write("{:.8f}\n".format(batch_cost))
    if (e+1) % display_step == 0:
        print("Epoch: {}/{}...".format(e+1, epochs),
                  "Training loss: {:.4f}".format(batch_cost),
                 "Validation loss: {:.4f}".format(batch_cost_test))   
    loss.append(batch_cost)
    valid_loss.append(batch_cost_test)
    plt.plot(range(e+1), loss, 'bo', label='Training loss')
    plt.plot(range(e+1), valid_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()
fo1.close()
fo = open("Real_High_Second_3_shortcuts.txt", "w")
sum_loss=0
for counter in range(100): 
    x_test_noisy,x_test_img=sess.run(next_patch2)
    test_imgs=x_test_img.reshape((-1, 512, 512, 1))
    x_test_noisy=x_test_noisy.reshape((-1, 512, 512, 1))
    recon_img = sess.run([decoded], feed_dict={inputs_: x_test_noisy})[0]
    MSR_loss=0
#    for index_test in range(63):
    noised_input = [cv2.cvtColor(cv2.resize(img,(224,224)),cv2.COLOR_GRAY2BGR) for img in recon_img]
    standard_input = [cv2.cvtColor(cv2.resize(img,(224,224)),cv2.COLOR_GRAY2BGR) for img in test_imgs]  
    content_loss2=sess.run(content_loss,feed_dict={image_noised:noised_input,image_standard:standard_input}) 
    MSR_loss=MSR_loss+content_loss2.sum()
    
#    fo.write("test round {}'s".format(counter))
    fo.write("{:.8f}\n".format(MSR_loss))
    sum_loss=sum_loss+MSR_loss
fo.write("{:.8f}\n".format(sum_loss)) 
fo.close()

for i in range(100):
    x_test_noisy,x_test_img=sess.run(next_patch4)
    test_imgs=x_test_img.reshape((-1, 512, 512, 1))
    
    x_test_noisy=x_test_noisy.reshape((-1, 512, 512, 1))
    recon_img = sess.run([decoded], feed_dict={inputs_: x_test_noisy})[0]
    for j in range(4):
        cv2.imwrite('Second High val-3shortcut/noisedimage/'+str(i*4+j).rjust(6,'0')+'.png',x_test_noisy[j]*255.0)
        cv2.imwrite('Second High val-3shortcut/targetimage/'+str(i*4+j).rjust(6,'0')+'.png',test_imgs[j]*255.0)
        cv2.imwrite('Second High val-3shortcut/resultimage/'+str(i*4+j).rjust(6,'0')+'.png',recon_img[j]*255.0)
saver.save(sess, 'Models/Real_High_second_high_3_Shortcuts_encode_model')  
writer.close()
 
sess.close()
    


