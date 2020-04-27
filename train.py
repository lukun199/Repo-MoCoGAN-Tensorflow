#coding:utf-8
# 'beta1_power' and 'beta2_power' are the params in Adam
import os, time, glob
import numpy as np
import tensorflow as tf
from models import GAN_net
from utils import *

DEPTH = 16
OUTPUT_SIZE = 64
batch_size = 8
gen_length = 16
channel = 3
EPOCH = 71

OS = 'LINUX' # optional: 'LINUX', 'WINDOWS'
sample_path = './img/'
ckpt_path = './checkpoint_video/'
moco_path = './actions/'

#------------------------------test----moco
import cv2
# index all images
list_of_images_folder = glob.glob(moco_path + '*')
list_of_images = []
list_of_images_length = []
list_of_images_category = []
for i, content in enumerate(list_of_images_folder):
    isfolder = content.split('/')[-1] if OS =='LINUX' else content.split('\\')[-1]
    if '.' not in isfolder:  # non-folders
        # enter sub-folders
        for j, image_name in enumerate(glob.glob(content + '/*')):
            img = cv2.imread(image_name)
            length_this = img.shape[1]//img.shape[0]
            list_of_images.append(img)
            list_of_images_length.append(length_this)
            list_of_images_category.append(int(isfolder))
print('length of deatset:', len(list_of_images_length))
print(length_this,'------------------dataset loaded--------------------[*]')

def get_data_batch_moco(batch_size,length_of_frame,height=64,width=64,channel=3,intervial=2):
    video = np.zeros([batch_size,length_of_frame,height,width,channel])
    label = np.zeros([batch_size, 4])
    for count, batch_this in enumerate(np.random.permutation(len(list_of_images_length))[:batch_size]):
        label[count,:] = np.eye(4)[list_of_images_category[batch_this]]
        length_this = list_of_images_length[batch_this]
        start = np.random.randint(length_this - intervial * length_of_frame - 1)
        for frame_this in range(length_of_frame):
            video[count,frame_this,:] = list_of_images[batch_this][:,start*height:height+start*height,:]
            start += intervial
    return image_to_pn_1(video), label
#------------------------------test

def train():
    gan_model = GAN_net(depth=DEPTH,output_size=OUTPUT_SIZE,batch_size=batch_size,gen_length=gen_length,channel=channel)
    real_data = tf.placeholder(tf.float32, shape=[batch_size, gen_length, OUTPUT_SIZE, OUTPUT_SIZE, channel])
    noise_coefficient = tf.placeholder(tf.float32, shape=[1,1])
    loss_pack, gen_train_op, disc_f_train_op, disc_v_train_op = gan_model.build_gan(real_data, noise_coefficient)

    saver = tf.train.Saver(max_to_keep=10)

    sess = tf.InteractiveSession()
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    init = tf.global_variables_initializer()
    sess.run(init)

    noise_weight = 1.0 * np.identity(1)  # feed_dict is problematic
    for epoch in range (1, EPOCH):
        total_inner_batch = 1601
        for iters in range(total_inner_batch):

            train_gif, text_real_onehot = get_data_batch_moco(batch_size,gen_length,OUTPUT_SIZE,OUTPUT_SIZE,channel)

            feed_dict = {real_data: train_gif, noise_coefficient: noise_weight}

            _, _, loss_detail = sess.run([disc_f_train_op, disc_v_train_op, loss_pack], feed_dict=feed_dict)
            _ = sess.run([gen_train_op], feed_dict=feed_dict)
            
            (g_loss_frame_isreal, g_loss_video_isreal, d_loss_frame_isreal, d_loss_video_isreal) = loss_detail

            if iters % 30 == 0:
                print("[epoch%4d: iter%4d/%4d]" % (epoch, iters, total_inner_batch))
                print("gf_isr=%.3f  gv_isr=%.3f" % (g_loss_frame_isreal, g_loss_video_isreal))
                print("df_isr=%.3f  dv_isr=%.3f\n" % (d_loss_frame_isreal, d_loss_video_isreal))

        if epoch % 5 == 0:
            saver.save(sess, ckpt_path + 'MOCO_REPO_lukun199')
            print('*********    model saved    *********')

        with tf.variable_scope(tf.get_variable_scope()):  # 可！
            samples = gan_model.test_gan()
            samples=sess.run(samples)
            save_images(samples, [4,4], channel, sample_path+'sample_epoch%3d.png' % epoch)

    sess.close()

if __name__ == '__main__':
    train()
