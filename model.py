import tensorflow as tf
import layers
from network import Generator
from network import Discriminator as Discriminator

DATA_FORMAT = 'NCHW'

class Model:
    def __init__(self, config=None):
        self.gan_type = 'wgan'
        # format parameters
        self.dtype = tf.float32
        self.data_format = DATA_FORMAT
        self.data_format_k = 'channels_first' if self.data_format == 'NCHW' else 'channels_last'
        self.in_channels = 3
        self.num_domains = None
        # collections
        self.g_train_sums = []
        self.d_train_sums = []
        self.loss_sums = []
        # copy all the properties from config object
        self.config = config
        if config is not None:
            self.__dict__.update(config.__dict__)
        # internal parameters
        self.input_shape = [None, None, None, None]
        self.input_shape[-3 if self.data_format == 'NCHW' else -1] = self.in_channels
        self.output_shape = self.input_shape
        self.domain_shape = [None, self.num_domains]
        self.domain_dtype = tf.int64

    def build_model(self, inputs=None, target_domains=None):
        # inputs
        if inputs is None:
            self.inputs = tf.placeholder(self.dtype, self.input_shape, name='Input')
        else:
            self.inputs = tf.identity(inputs, name='Input')
            self.inputs.set_shape(self.input_shape)
        # target domains
        if target_domains is None:
            self.target_domains = tf.placeholder(self.domain_dtype, self.domain_shape, name='Domain')
        else:
            self.target_domains = tf.identity(target_domains, name='Domain')
            self.target_domains.set_shape(self.domain_shape)
        # forward pass
        self.generator = Generator('Generator', self.config)
        self.outputs = self.generator(self.inputs, self.target_domains, reuse=None)
        # outputs
        self.outputs = tf.identity(self.outputs, name='Output')
        # all the saver variables
        self.svars = self.generator.svars
        # all the restore variables
        self.rvars = self.generator.rvars
        # return outputs
        return self.outputs

    def build_train(self, inputs=None, origin_domains=None, target_domains=None):
        # origin domains
        if origin_domains is None:
            self.origin_domains = tf.placeholder(self.domain_dtype, self.domain_shape, name='OriginDomain')
        else:
            self.origin_domains = tf.identity(origin_domains, name='OriginDomain')
            self.origin_domains.set_shape(self.domain_shape)
        # build model
        self.build_model(inputs, target_domains)
        # reconstruction
        self.reconstructs = self.generator(self.outputs, self.origin_domains, reuse=True)
        # discrimination
        self.discriminator = Discriminator('Discriminator', self.config)
        critic_logit, domain_logit = self.discriminator(self.outputs, reuse=None)
        # build loss
        self.build_g_loss(self.inputs, self.outputs, self.reconstructs,
            self.target_domains, critic_logit, domain_logit)
        self.build_d_loss(self.inputs, self.origin_domains, self.outputs, critic_logit)

    def build_g_loss(self, inputs, outputs, reconstructs,
        target_domains, critic_logit, domain_logit):
        self.g_log_losses = []
        update_ops = []
        loss_key = 'GeneratorLoss'
        with tf.variable_scope(loss_key):
            # adversarial loss
            adv_loss = layers.GeneratorLoss(critic_logit, self.gan_type)
            tf.losses.add_loss(adv_loss)
            update_ops.append(self.loss_summary('adv_loss', adv_loss, self.g_log_losses))
            # domain classification loss
            cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(target_domains, tf.float32), logits=domain_logit))
            tf.losses.add_loss(cls_loss)
            update_ops.append(self.loss_summary('cls_loss', cls_loss, self.g_log_losses))
            # reconstruction loss
            rec_loss = tf.losses.absolute_difference(inputs, reconstructs, weights=1.0,
                loss_collection=None)
            '''
            ssim_loss = tf.constant(0, tf.float32)
            for _input, _recon in zip(tf.split(inputs, self.in_channels, axis=-3),
                tf.split(reconstructs, self.in_channels, axis=-3)):
                ssim_loss += 1 - layers.MS_SSIM(_input + 1, _recon + 1, L=2,
                    data_format=self.data_format, one_dim=False)
            rec_loss = ssim_loss
            '''
            tf.losses.add_loss(rec_loss)
            update_ops.append(self.loss_summary('rec_loss', rec_loss, self.g_log_losses))
            # total loss
            losses = tf.losses.get_losses(loss_key)
            main_loss = tf.add_n(losses, 'main_loss')
            # final loss
            self.g_loss = main_loss
            update_ops.append(self.loss_summary('loss', self.g_loss))
            # accumulate operator
            with tf.control_dependencies(update_ops):
                self.g_losses_acc = tf.no_op('accumulator')

    def build_d_loss(self, real, real_domain_label, fake, fake_critic):
        self.d_log_losses = []
        update_ops = []
        loss_key = 'DiscriminatorLoss'
        real_critic, real_domain_logit = self.discriminator(real, reuse=True)
        # WGAN lipschitz-penalty
        use_gp = 'wgan' in self.gan_type or self.gan_type == 'dragan'
        def random_interpolate(gan_type):
            shape = tf.shape(real)
            batch_shape = shape * [1, 0, 0, 0] + [0, 1, 1, 1]
            if gan_type == 'dragan':
                eps = tf.random_uniform(shape, minval=0., maxval=1.)
                x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
                x_std = tf.sqrt(x_var)
                noise = 0.5 * x_std * eps
                alpha = tf.random_uniform(batch_shape, minval=-1., maxval=1.)
                alpha.set_shape([None, self.in_channels, None, None])
                interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)
            else:
                alpha = tf.random_uniform(batch_shape, minval=0., maxval=1.)
                alpha.set_shape([None, self.in_channels, None, None])
                differences = fake - real
                interpolated = alpha * differences + real
            return interpolated
        if use_gp and self.gan_type != 'wgan-div':
            inter = random_interpolate(self.gan_type)
            inter_critic, inter_domain = self.discriminator(inter, reuse=True)
        with tf.variable_scope(loss_key):
            # adversarial loss
            adv_loss = layers.DiscriminatorLoss(real_critic, fake_critic, self.gan_type)
            tf.losses.add_loss(adv_loss)
            update_ops.append(self.loss_summary('adv_loss', adv_loss, self.d_log_losses))
            # gradient loss
            if use_gp:
                flatten = tf.layers.Flatten(self.data_format_k)
                if self.gan_type == 'wgan-div':
                    k = 2
                    p = 6
                    real_grad = tf.gradients(real_critic, [real])[0]
                    real_grad_norm = tf.reduce_sum(tf.square(flatten(real_grad)), axis=1) ** (p / 2)
                    grad_loss = tf.reduce_mean(real_grad_norm) * k
                else:
                    lambda_gp = 10
                    K = 1.0
                    inter_grad = tf.gradients(inter_critic, [inter])[0]
                    inter_grad_norm = tf.norm(flatten(inter_grad), axis=1)
                    grad_loss = lambda_gp * tf.reduce_mean(tf.square(inter_grad_norm - K))
                tf.losses.add_loss(grad_loss)
                update_ops.append(self.loss_summary('grad_loss', grad_loss, self.d_log_losses))
            # domain classification loss
            cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(real_domain_label, tf.float32), logits=real_domain_logit))
            tf.losses.add_loss(cls_loss)
            update_ops.append(self.loss_summary('cls_loss', cls_loss, self.d_log_losses))
            # total loss
            losses = tf.losses.get_losses(loss_key)
            main_loss = tf.add_n(losses, 'main_loss')
            # final loss
            self.d_loss = main_loss
            update_ops.append(self.loss_summary('loss', self.d_loss))
            # accumulate operator
            with tf.control_dependencies(update_ops):
                self.d_losses_acc = tf.no_op('accumulator')

    def train_g(self, global_step):
        model = self.generator
        # dependencies to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Generator')
        # learning rate
        lr_base = 2e-5
        lr = 2 * lr_base / self.config.max_steps * (
            1.0 * self.config.max_steps - tf.cast(global_step, tf.float32))
        lr = tf.clip_by_value(lr, lr_base * 0, lr_base)
        self.g_train_sums.append(tf.summary.scalar('Generator/LR', lr))
        # optimizer
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
        with tf.control_dependencies(update_ops):
            grads_vars = opt.compute_gradients(self.g_loss, model.tvars)
            update_ops = [opt.apply_gradients(grads_vars, global_step)]
        # histogram for gradients and variables
        for grad, var in grads_vars:
            self.g_train_sums.append(tf.summary.histogram(var.op.name + '/grad', grad))
            self.g_train_sums.append(tf.summary.histogram(var.op.name, var))
        # save moving average of trainalbe variables
        update_ops = model.apply_ema(update_ops)
        # all the saver variables
        self.svars = model.svars
        # return optimizing op
        with tf.control_dependencies(update_ops):
            train_op = tf.no_op('train_g')
        return train_op

    def train_d(self, global_step):
        model = self.discriminator
        # dependencies to be updated
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Discriminator')
        # learning rate
        lr_base = 2e-5
        lr = 2 * lr_base / self.config.max_steps * (
            1.0 * self.config.max_steps - tf.cast(global_step, tf.float32))
        lr = tf.clip_by_value(lr, lr_base * 0, lr_base)
        self.d_train_sums.append(tf.summary.scalar('Discriminator/LR', lr))
        # optimizer
        opt = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
        with tf.control_dependencies(update_ops):
            grads_vars = opt.compute_gradients(self.d_loss, model.tvars)
            update_ops = [opt.apply_gradients(grads_vars)]
        # histogram for gradients and variables
        for grad, var in grads_vars:
            self.d_train_sums.append(tf.summary.histogram(var.op.name + '/grad', grad))
            self.d_train_sums.append(tf.summary.histogram(var.op.name, var))
        # return optimizing op
        with tf.control_dependencies(update_ops):
            train_op = tf.no_op('train_d')
        return train_op

    def loss_summary(self, name, loss, collection=None):
        with tf.variable_scope('LossSummary/' + name):
            # internal variables
            loss_sum = tf.get_variable('sum', (), tf.float32, tf.initializers.zeros(tf.float32))
            loss_count = tf.get_variable('count', (), tf.float32, tf.initializers.zeros(tf.float32))
            # accumulate to sum and count
            acc_sum = loss_sum.assign_add(loss, True)
            acc_count = loss_count.assign_add(1.0, True)
            # calculate mean
            loss_mean = tf.divide(loss_sum, loss_count, 'mean')
            if collection is not None:
                collection.append(loss_mean)
            # reset sum and count
            with tf.control_dependencies([loss_mean]):
                clear_sum = loss_sum.assign(0.0, True)
                clear_count = loss_count.assign(0.0, True)
            # log summary
            with tf.control_dependencies([clear_sum, clear_count]):
                self.loss_sums.append(tf.summary.scalar('value', loss_mean))
            # return after updating sum and count
            with tf.control_dependencies([acc_sum, acc_count]):
                return tf.identity(loss, 'loss')

    def get_summaries(self):
        g_train_summary = tf.summary.merge(self.g_train_sums) if self.g_train_sums else None
        d_train_summary = tf.summary.merge(self.d_train_sums) if self.d_train_sums else None
        loss_summary = tf.summary.merge(self.loss_sums) if self.loss_sums else None
        return g_train_summary, d_train_summary, loss_summary
