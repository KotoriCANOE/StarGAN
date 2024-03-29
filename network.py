import tensorflow as tf
import tensorflow.contrib.layers as slim
import layers

ACTIVATION = layers.Swish
DATA_FORMAT = 'NCHW'

# Generator

class GeneratorConfig:
    def __init__(self):
        # format parameters
        self.dtype = tf.float32
        self.data_format = DATA_FORMAT
        self.in_channels = 3
        self.num_domains = None
        # model parameters
        self.activation = ACTIVATION
        self.normalization = 'Instance'
        # train parameters
        self.random_seed = 0
        self.var_ema = 0.999
        self.weight_decay = 1e-6

class Generator(GeneratorConfig):
    def __init__(self, name='Generator', config=None):
        super().__init__()
        self.name = name
        # copy all the properties from config object
        if config is not None:
            self.__dict__.update(config.__dict__)
        # create a moving average object for trainable variables
        if self.var_ema > 0:
            self.ema = tf.train.ExponentialMovingAverage(self.var_ema)

    def ResBlock(self, last, channels, kernel=[3, 3], stride=[1, 1], biases=True, format=DATA_FORMAT,
        dilate=1, activation=ACTIVATION, normalizer=None,
        regularizer=None, collections=None):
        biases = tf.initializers.zeros(self.dtype) if biases else None
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        # pre-activation
        if normalizer: last = normalizer(last)
        if activation: last = activation(last)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, activation, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, None, None, None, initializer, regularizer, biases,
            variables_collections=collections)
        # residual connection
        last = layers.SEUnit(last, channels, format, regularizer, collections)
        last += skip
        return last

    def EBlock(self, last, channels, resblocks=1,
        kernel=[4, 4], stride=[2, 2], format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        # pre-activation
        if activation: last = activation(last)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        # residual blocks
        for i in range(resblocks):
            with tf.variable_scope('ResBlock_{}'.format(i)):
                last = self.ResBlock(last, channels, format=format,
                    activation=activation, normalizer=normalizer,
                    regularizer=regularizer, collections=collections)
        return last

    def DBlock(self, last, channels, resblocks=1,
        kernel=[3, 3], stride=[2, 2], format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        # pre-activation
        if activation: last = activation(last)
        # upsample
        with tf.variable_scope('Upsample'):
            upsize = tf.shape(last)
            upsize = upsize[-2:] if format == 'NCHW' else upsize[-3:-1]
            upsize = upsize * stride[0:2]
            if format == 'NCHW':
                last = tf.transpose(last, (0, 2, 3, 1))
            last = tf.image.resize_nearest_neighbor(last, upsize)
            if format == 'NCHW':
                last = tf.transpose(last, (0, 3, 1, 2))
        # convolution
        last = slim.conv2d(last, channels, kernel, [1, 1], 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        # residual blocks
        for i in range(resblocks):
            with tf.variable_scope('ResBlock_{}'.format(i)):
                last = self.ResBlock(last, channels, format=format,
                    activation=activation, normalizer=normalizer,
                    regularizer=regularizer, collections=collections)
        return last

    def __call__(self, last, domains, reuse=None):
        format = self.data_format
        channel_index = -3 if format == 'NCHW' else -1
        kernel1 = [4, 4]
        stride1 = [2, 2]
        kernel2 = [3, 3]
        stride2 = [2, 2]
        # function objects
        activation = self.activation
        if self.normalization == 'Batch':
            normalizer = lambda x: slim.batch_norm(x, 0.999, center=True, scale=True,
                is_training=self.training, data_format=format, renorm=False)
        elif self.normalization == 'Instance':
            normalizer = lambda x: slim.instance_norm(x, center=True, scale=True, data_format=format)
        elif self.normalization == 'Group':
            normalizer = lambda x: (slim.group_norm(x, x.shape.as_list()[-3] // 16, -3, (-2, -1))
                if format == 'NCHW' else slim.group_norm(x, x.shape.as_list()[-1] // 16, -1, (-3, -2)))
        else:
            normalizer = None
        regularizer = slim.l2_regularizer(self.weight_decay) if self.weight_decay else None
        skip_connection = lambda x, y: x + y
        # skip_connection = lambda x, y: tf.concat([x, y], channel_index)
        # model scope
        with tf.variable_scope(self.name, reuse=reuse):
            # states
            self.training = tf.Variable(False, trainable=False, name='training',
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
            skips = []
            # concat
            # domains = tf.one_hot(domains, self.num_domains)
            domains = tf.cast(domains, tf.float32)
            domains = tf.reshape(domains, (-1, self.num_domains, 1, 1))
            domains_shape = tf.shape(last)
            domains_shape = domains_shape * [1, 0, 1, 1] + [0, self.num_domains, 0, 0]
            domains = tf.broadcast_to(domains, domains_shape)
            domains_shape = [None] * 4
            domains_shape[channel_index] = self.num_domains
            domains.set_shape(domains_shape)
            last = tf.concat([last, domains], channel_index)
            # encoder
            with tf.variable_scope('InBlock'):
                last = self.EBlock(last, 16, 0, [7, 7], [1, 1],
                    format, None, None, regularizer)
            with tf.variable_scope('EBlock_0'):
                skips.append(last)
                last = self.EBlock(last, 32, 0, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_1'):
                skips.append(last)
                last = self.EBlock(last, 64, 1, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_2'):
                skips.append(last)
                last = self.EBlock(last, 128, 1, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_3'):
                skips.append(last)
                last = self.EBlock(last, 192, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_4'):
                skips.append(last)
                last = self.EBlock(last, 256, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            # decoder
            with tf.variable_scope('DBlock_4'):
                last = self.DBlock(last, 192, 2, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_3'):
                last = self.DBlock(last, 128, 1, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_2'):
                last = self.DBlock(last, 64, 1, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_1'):
                last = self.DBlock(last, 32, 0, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('DBlock_0'):
                last = self.DBlock(last, 16, 0, kernel2, stride2,
                    format, activation, normalizer, regularizer)
                last = skip_connection(last, skips.pop())
            with tf.variable_scope('OutBlock'):
                last = self.EBlock(last, self.in_channels, 0, [7, 7], [1, 1],
                    format, activation, normalizer, regularizer)
        # trainable/model/save/restore variables
        self.tvars = tf.trainable_variables(self.name)
        self.mvars = tf.model_variables(self.name)
        self.mvars = [i for i in self.mvars if i not in self.tvars]
        self.svars = list(set(self.tvars + self.mvars))
        self.rvars = self.svars.copy()
        # restore moving average of trainable variables
        if self.var_ema > 0:
            with tf.variable_scope('EMA'):
                self.rvars = {**{self.ema.average_name(var): var for var in self.tvars},
                    **{var.op.name: var for var in self.mvars}}
        return last

    def apply_ema(self, update_ops=[]):
        if not self.var_ema:
            return update_ops
        with tf.variable_scope('EMA'):
            with tf.control_dependencies(update_ops):
                update_ops = [self.ema.apply(self.tvars)]
            self.svars = [self.ema.average(var) for var in self.tvars] + self.mvars
        return update_ops

# Discriminator

class DiscriminatorConfig:
    def __init__(self):
        # format parameters
        self.dtype = tf.float32
        self.data_format = DATA_FORMAT
        self.in_channels = 3
        self.num_domains = None
        # model parameters
        self.activation = ACTIVATION
        self.normalization = None
        # train parameters
        self.random_seed = 0
        self.weight_decay = 1e-6

class Discriminator(DiscriminatorConfig):
    def __init__(self, name='Discriminator', config=None):
        super().__init__()
        self.name = name
        # copy all the properties from config object
        if config is not None:
            self.__dict__.update(config.__dict__)

    def ResBlock(self, last, channels, kernel=[3, 3], stride=[1, 1], biases=True, format=DATA_FORMAT,
        dilate=1, activation=ACTIVATION, normalizer=None,
        regularizer=None, collections=None):
        biases = tf.initializers.zeros(self.dtype) if biases else None
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        # pre-activation
        if normalizer: last = normalizer(last)
        if activation: last = activation(last)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, activation, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, None, None, None, initializer, regularizer, biases,
            variables_collections=collections)
        # residual connection
        last = layers.SEUnit(last, channels, format, regularizer, collections)
        last += skip
        return last

    def InBlock(self, last, channels, kernel=[3, 3], stride=[1, 1], format=DATA_FORMAT,
        activation=None, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, activation, normalizer, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        return last

    def EBlock(self, last, channels, resblocks=1,
        kernel=[4, 4], stride=[2, 2], format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        # pre-activation
        if activation: last = activation(last)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, None, None, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        # residual blocks
        for i in range(resblocks):
            with tf.variable_scope('ResBlock_{}'.format(i)):
                last = self.ResBlock(last, channels, format=format,
                    activation=activation, normalizer=normalizer,
                    regularizer=regularizer, collections=collections)
        # dense connection
        with tf.variable_scope('DenseConnection'):
            last = layers.SEUnit(last, channels, format, regularizer, collections)
            if stride != 1 or stride != [1, 1]:
                pool_stride = [1, 1] + stride if format == 'NCHW' else [1] + stride + [1]
                skip = tf.nn.avg_pool(skip, pool_stride, pool_stride, 'SAME', format)
            last = tf.concat([skip, last], -3 if format == 'NCHW' else -1)
        return last

    def __call__(self, last, reuse=None):
        # parameters
        format = self.data_format
        kernel1 = [4, 4]
        stride1 = [2, 2]
        # function objects
        activation = self.activation
        if self.normalization == 'Batch':
            normalizer = lambda x: slim.batch_norm(x, 0.999, center=True, scale=True,
                is_training=self.training, data_format=format, renorm=False)
        elif self.normalization == 'Instance':
            normalizer = lambda x: slim.instance_norm(x, center=True, scale=True, data_format=format)
        elif self.normalization == 'Group':
            normalizer = lambda x: (slim.group_norm(x, x.shape.as_list()[-3] // 16, -3, (-2, -1))
                if format == 'NCHW' else slim.group_norm(x, x.shape.as_list()[-1] // 16, -1, (-3, -2)))
        else:
            normalizer = None
        regularizer = slim.l2_regularizer(self.weight_decay) if self.weight_decay else None
        # model scope
        with tf.variable_scope(self.name, reuse=reuse):
            # states
            self.training = tf.Variable(False, trainable=False, name='training',
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
            # encoder
            with tf.variable_scope('InBlock'):
                last = self.InBlock(last, 32, [7, 7], [1, 1],
                    format, None, None, regularizer)
            with tf.variable_scope('EBlock_1'):
                last = self.EBlock(last, 32, 0, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_2'):
                last = self.EBlock(last, 32, 1, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_3'):
                last = self.EBlock(last, 40, 1, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_4'):
                last = self.EBlock(last, 40, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_5'):
                last = self.EBlock(last, 40, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_6'):
                last = self.EBlock(last, 40, 2, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('PatchCritic'):
                last_channels = last.shape.as_list()[-3]
                critic_logit = self.ResBlock(last, last_channels, [3, 3], [1, 1], format=format,
                    activation=activation, normalizer=normalizer, regularizer=regularizer)
                critic_logit = slim.conv2d(critic_logit, 1, [7, 7], [1, 1], 'SAME', format,
                    1, None, None, weights_regularizer=regularizer)
            with tf.variable_scope('GlobalAveragePooling'):
                last = tf.reduce_mean(last, [-2, -1] if format == 'NCHW' else [-3, -2])
            with tf.variable_scope('Domain'):
                last_channels = last.shape.as_list()[-1]
                domain_logit = slim.fully_connected(last, last_channels, activation, None,
                    weights_regularizer=regularizer)
                domain_logit = slim.fully_connected(domain_logit, self.num_domains, None, None,
                    weights_regularizer=regularizer)
        # trainable/model/save/restore variables
        self.tvars = tf.trainable_variables(self.name)
        self.mvars = tf.model_variables(self.name)
        self.mvars = [i for i in self.mvars if i not in self.tvars]
        self.svars = list(set(self.tvars + self.mvars))
        self.rvars = self.svars.copy()
        return critic_logit, domain_logit

# Discriminator 2

class Discriminator2(DiscriminatorConfig):
    def __init__(self, name='Discriminator', config=None):
        super().__init__()
        self.name = name
        # copy all the properties from config object
        if config is not None:
            self.__dict__.update(config.__dict__)

    def ResBlock(self, last, channels, kernel=[3, 3], stride=[1, 1], biases=True, format=DATA_FORMAT,
        dilate=1, activation=ACTIVATION, normalizer=None,
        regularizer=None, collections=None):
        biases = tf.initializers.zeros(self.dtype) if biases else None
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        skip = last
        # pre-activation
        if normalizer: last = normalizer(last)
        if activation: last = activation(last)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, activation, normalizer, None, initializer, regularizer, biases,
            variables_collections=collections)
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            dilate, None, None, None, initializer, regularizer, biases,
            variables_collections=collections)
        # residual connection
        last = layers.SEUnit(last, channels, format, regularizer, collections)
        last += skip
        return last

    def InBlock(self, last, channels, kernel=[3, 3], stride=[1, 1], format=DATA_FORMAT,
        activation=None, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        # convolution
        last = slim.conv2d(last, channels, kernel, stride, 'SAME', format,
            1, activation, normalizer, weights_initializer=initializer,
            weights_regularizer=regularizer, variables_collections=collections)
        return last

    def EBlock(self, last, l=2, channels=None,
        kernel=[3, 3], stride=[2, 2], format=DATA_FORMAT,
        activation=ACTIVATION, normalizer=None, regularizer=None, collections=None):
        initializer = tf.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', self.random_seed, self.dtype)
        channel_index = -3 if format == 'NCHW' else -1
        last_channels = last.shape.as_list()[channel_index]
        # dense layers
        for _ in range(l):
            if channels is not None and _ == l - 1:
                last_channels = channels
            # convolution
            last = slim.conv2d(last, last_channels, kernel, [1, 1], 'SAME', format,
                1, None, None, weights_initializer=initializer,
                weights_regularizer=regularizer, variables_collections=collections)
            # post-activation
            if normalizer: last = normalizer(last)
            if activation: last = activation(last)
        # pooling
        if stride and stride != [1, 1] and stride != 1:
            strides = [1, 1] + stride if format == 'NCHW' else [1] + stride + [1]
            last = tf.nn.avg_pool(last, strides, strides, 'SAME', format)
        # return
        return last

    def __call__(self, last, reuse=None):
        # parameters
        format = self.data_format
        kernel1 = [3, 3]
        stride1 = [2, 2]
        # function objects
        activation = self.activation
        if self.normalization == 'Batch':
            normalizer = lambda x: slim.batch_norm(x, 0.999, center=True, scale=True,
                is_training=self.training, data_format=format, renorm=False)
        elif self.normalization == 'Instance':
            normalizer = lambda x: slim.instance_norm(x, center=True, scale=True, data_format=format)
        elif self.normalization == 'Group':
            normalizer = lambda x: (slim.group_norm(x, x.shape.as_list()[-3] // 16, -3, (-2, -1))
                if format == 'NCHW' else slim.group_norm(x, x.shape.as_list()[-1] // 16, -1, (-3, -2)))
        else:
            normalizer = None
        regularizer = slim.l2_regularizer(self.weight_decay) if self.weight_decay else None
        # model scope
        with tf.variable_scope(self.name, reuse=reuse):
            # states
            self.training = tf.Variable(False, trainable=False, name='training',
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
            # encoder
            with tf.variable_scope('InBlock'):
                last = self.InBlock(last, 32, [7, 7], [1, 1],
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_1'):
                last = self.EBlock(last, 1, 48, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_2'):
                last = self.EBlock(last, 1, 64, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_3'):
                last = self.EBlock(last, 1, 96, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_4'):
                last = self.EBlock(last, 1, 128, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_5'):
                last = self.EBlock(last, 1, 192, kernel1, stride1,
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('EBlock_6'):
                last = self.EBlock(last, 1, 256, kernel1, [1, 1],
                    format, activation, normalizer, regularizer)
            with tf.variable_scope('PatchCritic'):
                last_channels = last.shape.as_list()[-3]
                critic_logit = self.ResBlock(last, last_channels, [3, 3], [1, 1], format=format,
                    activation=activation, normalizer=normalizer, regularizer=regularizer)
                critic_logit = slim.conv2d(critic_logit, 1, [7, 7], [1, 1], 'SAME', format,
                    1, None, None, weights_regularizer=regularizer)
            with tf.variable_scope('GlobalAveragePooling'):
                last = tf.reduce_mean(last, [-2, -1] if format == 'NCHW' else [-3, -2])
            with tf.variable_scope('Domain'):
                last_channels = last.shape.as_list()[-1]
                domain_logit = slim.fully_connected(last, last_channels, activation, None,
                    weights_regularizer=regularizer)
                domain_logit = slim.fully_connected(domain_logit, self.num_domains, None, None,
                    weights_regularizer=regularizer)
        # trainable/model/save/restore variables
        self.tvars = tf.trainable_variables(self.name)
        self.mvars = tf.model_variables(self.name)
        self.mvars = [i for i in self.mvars if i not in self.tvars]
        self.svars = list(set(self.tvars + self.mvars))
        self.rvars = self.svars.copy()
        return critic_logit, domain_logit
