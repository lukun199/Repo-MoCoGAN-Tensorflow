import tensorflow as tf
from utils import *


class GAN_net():
    def __init__(self, depth=16, output_size=48, batch_size=16, gen_length=16, channel=3):
        self.depth = depth
        self.output_size = output_size
        self.batch_size = batch_size
        self.gen_length = gen_length
        self.channel = channel
        self.noise_content = 50
        self.noise_motion = 10
        self.latent_size = self.noise_content + self.noise_motion

    def frame_discriminator(self, name, inputs, coefficient, reuse=False, is_training=True):
        # we don't care the exact nums at this time. e.g. DEPTH, width, height, channels of input images
        # input: [batch_size,height,width,channel]
        # text: [batch_size, dim]
        # out: [batch_size, 1]
        use_noise = True

        with tf.variable_scope(name, reuse=reuse):  # in_dim = 48

            output1 = conv2d('d_frame_conv_1', inputs, ksize=5, out_dim=self.depth)  # dim = 32
            output2 = lrelu('d_frame_lrelu_1', addnoise(coefficient, output1, is_use=use_noise))

            output3 = conv2d('d_frame_conv_2', output2, ksize=5, out_dim=2 * self.depth)  # 16
            output3 = batch_norm('bn_d_frame_2', output3, is_training=is_training)
            output4 = lrelu('d_frame_lrelu_2', addnoise(coefficient, output3, is_use=use_noise))

            output5 = conv2d('d_frame_conv_3', output4, ksize=5, out_dim=4 * self.depth)  # 8
            output5 = batch_norm('bn_d_frame_3', output5, is_training=is_training)
            output6 = lrelu('d_frame_lrelu_3', addnoise(coefficient, output5, is_use=use_noise))

            output7 = conv2d('d_frame_conv_4', output6, ksize=5, out_dim=8 * self.depth)  # 4
            output7 = batch_norm('bn_d_frame_4', output7, is_training=is_training)
            output8 = lrelu('d_frame_lrelu_4', addnoise(coefficient, output7, is_use=use_noise))

            output8 = tf.reshape(output8, [inputs.get_shape().as_list()[0], -1])

            prob = fully_connected('d_frame_prob', output8, 1)


            return prob

    def video_discriminator(self, name, inputs, coefficient, reuse=False, is_training=True):
        # we don't care the exact nums at this time. e.g. DEPTH, width, height, channels of input images
        # input: [batch_size,length,height,width,channel]  text_video: [batch_size, dim_motion]
        # out: [batch_size, 1]
        use_noise = True

        with tf.variable_scope(name, reuse=reuse):
            output1 = conv3d('d_video_conv_1', inputs, ksize=5, out_dim=self.depth, pad_dim=2)  # 13
            output2 = lrelu('d_video_lrelu_1', addnoise(coefficient, output1, is_use=use_noise))

            output3 = conv3d('d_video_conv_2', output2, ksize=5, out_dim=2 * self.depth, pad_dim=2)
            output3 = batch_norm('bn_d_video_2', output3, is_training=is_training)
            output4 = lrelu('d_video_lrelu_2', addnoise(coefficient, output3, is_use=use_noise))

            output5 = conv3d('d_video_conv_3', output4, ksize=5, out_dim=4 * self.depth, pad_dim=2)
            output5 = batch_norm('bn_d_video_3', output5, is_training=is_training)
            output6 = lrelu('d_video_lrelu_3', addnoise(coefficient, output5, is_use=use_noise))

            output7 = conv3d('d_video_conv_4', output6, ksize=5, out_dim=8 * self.depth, pad_dim=2)
            output7 = batch_norm('bn_d_video_4', output7, is_training=is_training)
            output8 = lrelu('d_video_lrelu_4', addnoise(coefficient, output7, is_use=use_noise))

            output8 = tf.reshape(output8, [inputs.get_shape().as_list()[0], -1])

            prob = fully_connected('d_video_prob', output8, 1)  # temp

            return prob

    def latent_lib(self, name, reuse=False, sample=None):
        # inout should be a sentence embedding, but here we simply use the ont-hot category information.
        # The latent code is partially generated based on the input, which is shared by the triplet input.
        # LSTM ref: https://blog.csdn.net/u014595019/article/details/52759104
        # https://www.cnblogs.com/Lee-yl/p/10079408.html
        # https://blog.csdn.net/u013230189/article/details/82817181?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

        if not sample: sample = self.batch_size

        with tf.variable_scope(name, reuse=reuse):

            latent_content = tf.random_normal([sample, self.noise_content])

            # define the gru cell
            gru_cell = tf.nn.rnn_cell.GRUCell(self.noise_motion,
                                              kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                              bias_initializer=tf.zeros_initializer())
            init_state = gru_cell.zero_state(sample, dtype=tf.float32)

            # run.
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=gru_cell,
                                                         inputs=tf.random_normal(
                                                        [sample, self.gen_length,
                                                         self.noise_motion], 0, 1, dtype=tf.float32),
                                                         initial_state=init_state)

            latent_z = tf.concat([tf.tile(tf.expand_dims(latent_content, 1), [1, self.gen_length, 1])
                                     , rnn_outputs]
                                 , axis=2)

        return latent_z

    def generator(self, name, latent_z, reuse=False, is_training=True):
        # all with relu activation except for the last layer / tanh
        # input: noise per frame
        # output: images
        # attention, the first dim might change.

        true_batch = latent_z.get_shape().as_list()[0]

        with tf.variable_scope(name, reuse=reuse):

            output = fully_connected('g_fc_1', latent_z, 4 * 4 * 8 * self.depth)  # batch not change
            output = tf.reshape(output, [true_batch, 4, 4, 8 * self.depth], 'g_conv')
            output = relu('g_deconv_0_relu', batch_norm('bn_g_0', output, is_training=is_training))

            output = deconv2d('g_deconv_1', output, ksize=5, outshape=[true_batch, 8, 8, 4 * self.depth])
            output = relu('g_deconv_1_relu', batch_norm('bn_g_1', output, is_training=is_training))

            output = deconv2d('g_deconv_2', output, ksize=5, outshape=[true_batch, 16, 16, 2 * self.depth])
            output = relu('g_deconv_2_relu', batch_norm('bn_g_2', output, is_training=is_training))

            output = deconv2d('g_deconv_3', output, ksize=5, outshape=[true_batch, 32, 32, self.depth])
            output = relu('g_deconv_3_relu', batch_norm('bn_g_3', output, is_training=is_training))

            output = deconv2d('g_deconv_4', output, ksize=5,
                              outshape=[true_batch, self.output_size, self.output_size, self.channel])
            output = tanh('g_tanh', output)

            return output


    def build_gan(self, real_data, noise_coefficient):

        # get latent code
        latent_z = self.latent_lib('g_latent_lib', reuse=False)


        shape_z = latent_z.get_shape().as_list()  # .as_list() important --lukun199
        shape_data = real_data.get_shape().as_list()

        # sample, reshape, and then fed into generator
        gen_input =  tf.reshape(latent_z,[shape_z[0]*shape_z[1],shape_z[2]])

        with tf.variable_scope(tf.get_variable_scope()):

            # generate output shape is [batch*self.gen_length,h,w,channel] first frame and then video.
            fake_data = self.generator('gen', gen_input, reuse=False)

            frame_out_fake = fake_data
            real_video_sample = real_data
            video_out_fake = tf.reshape(fake_data, [self.batch_size, self.gen_length,
                                                    shape_data[2], shape_data[3],shape_data[4]])
            real_frame_sample = tf.reshape(real_data, [self.batch_size * self.gen_length,
                                                    shape_data[2], shape_data[3],shape_data[4]])

            # discriminate

            dis_reuse_flag = False
            real_frame_isreal = self.frame_discriminator('dis_frame', real_frame_sample, noise_coefficient, reuse=dis_reuse_flag)
            real_video_isreal = self.video_discriminator('dis_video', real_video_sample, noise_coefficient, reuse=dis_reuse_flag)
            dis_reuse_flag = True
            fake_frame_isreal = self.frame_discriminator('dis_frame', frame_out_fake, noise_coefficient, reuse=dis_reuse_flag)
            fake_video_isreal = self.video_discriminator('dis_video', video_out_fake, noise_coefficient,reuse=dis_reuse_flag)

        t_vars = tf.trainable_variables()
        df_vars = [var for var in t_vars if 'dis_frame' in var.name]
        dv_vars = [var for var in t_vars if 'dis_video' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]


        update_ops_DF = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'dis_frame' in var.name]
        update_ops_DV = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'dis_video' in var.name]
        update_ops_G = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'g_' in var.name]


        gf_prob = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_frame_isreal, labels=tf.ones_like(fake_frame_isreal)))
        gv_prob = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_video_isreal, labels=tf.ones_like(fake_video_isreal)))

        df_prob = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_frame_isreal, labels=tf.ones_like(real_frame_isreal))) \
                    + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_frame_isreal, labels=tf.zeros_like(fake_frame_isreal)))
        dv_prob = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_video_isreal, labels=tf.ones_like(real_video_isreal))) \
                    + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_video_isreal, labels=tf.zeros_like(fake_video_isreal)))


        gen_cost = gf_prob + gv_prob



        with tf.control_dependencies(update_ops_DF):
            disc_f_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(df_prob, var_list=df_vars)

        with tf.control_dependencies(update_ops_DV):
            disc_v_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(dv_prob, var_list=dv_vars)

        with tf.control_dependencies(update_ops_G):
            gen_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(gen_cost, var_list=g_vars)
        
        loss_pack = (gf_prob, gv_prob, df_prob, dv_prob)

        return loss_pack, gen_train_op, disc_f_train_op, disc_v_train_op  # , gradient_penalty

    def test_gan(self, reuse=True):

        latent_z = self.latent_lib('g_latent_lib', reuse=reuse, sample=1)
        shape = latent_z.get_shape().as_list()  # .as_list() mportant --lukun199
        # reshape and feed into the generator
        latent_z = tf.reshape(latent_z, [shape[0] * shape[1], shape[2]])
        # generate output shape is [batch_size*frames,h,w,channel]
        fake_data = self.generator('gen', latent_z, reuse=reuse, is_training=False)

        return fake_data
