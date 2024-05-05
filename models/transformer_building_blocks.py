from keras import layers, Layer
import numpy as np
import tensorflow as tf

class Time2Vector(Layer):
    def __init__(self, timesteps, **kwargs):
        super(Time2Vector, self).__init__()
        self.timesteps = timesteps

    def build(self, input_shape):
        '''Initialize weights and biases with shape (batch, timesteps)'''
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.timesteps),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.timesteps),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.timesteps),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.timesteps),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        '''Calculate linear and periodic time features'''
        x = tf.math.reduce_mean(x[:, :, :4], axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear  # Linear time feature
        time_linear = tf.expand_dims(time_linear, axis=-1)  # Add dimension (batch, timesteps, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)  # Add dimension (batch, timesteps, 1)
        return tf.concat([time_linear, time_periodic], axis=-1)  # shape = (batch, timesteps, 2)

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'timesteps': self.timesteps})
        return config

###################################################################################################

class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = layers.Dense(self.d_k,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

        self.key = layers.Dense(self.d_k,
                         input_shape=input_shape,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='glorot_uniform')

        self.value = layers.Dense(self.d_v,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

    def call(self, inputs, mask=False):
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)

        # apply causal self attention if mask=True
        if mask:
            t = inputs[0].shape[1]
            look_ahead_mask = np.tril(np.ones((t, t)))
            attn_weights += look_ahead_mask * -1e9
        attn_weights = tf.map_fn(lambda x: x / np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out

###################################################################################################

class MultiAttention(Layer):
    def __init__(self, d_k, d_v, d_model, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for i in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))

        # apply fc layer to project data back into original d_model so dims match when applying residual connection
        self.linear = layers.Dense(self.d_model,
                            input_shape=input_shape,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='glorot_uniform')

    def call(self, inputs, mask=False):
        attn = [self.attn_heads[i](inputs, mask=mask) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear

###################################################################################################

class Encoder(Layer):
    def __init__(self, d_k, d_v, n_heads, d_ff, d_model, dropout=0.1, **kwargs):
        super(Encoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.d_model = d_model
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.d_model, self.n_heads)
        self.attn_dropout = layers.Dropout(self.dropout_rate)
        self.attn_normalize = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = layers.Conv1D(filters=self.d_ff, kernel_size=1, activation='relu')
        self.ff_conv1D_2 = layers.Conv1D(filters=self.d_model, kernel_size=1)
        self.ff_dropout = layers.Dropout(self.dropout_rate)
        self.ff_normalize = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        attn_output = self.attn_multi(inputs)
        attn_output = self.attn_dropout(attn_output)
        attn_output = self.attn_normalize(inputs[0] + attn_output)

        ff_output = self.ff_conv1D_1(attn_output)
        ff_output = self.ff_conv1D_2(ff_output)
        ff_output = self.ff_dropout(ff_output)
        ff_output = self.ff_normalize(inputs[0] + ff_output)
        return ff_output

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'd_ff': self.d_ff,
                       'd_model': self.d_model,
                       'attn_heads': self.attn_heads,
                       'dropout_rate': self.dropout_rate})
        return config

###################################################################################################

class Decoder(Layer):
    def __init__(self, d_k, d_v, n_heads, d_ff, d_model, dropout=0.1, **kwargs):
        super(Decoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.d_model = d_model
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.masked_attn_multi = MultiAttention(self.d_k, self.d_v, self.d_model, self.n_heads)
        self.attn_normalize = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_ff
                                  , activation='relu'),
            tf.keras.layers.Dense(self.d_model)
        ])

        self.ff_dropout = layers.Dropout(self.dropout_rate)
        self.ff_normalize = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs):
        # apply masked causal attention
        attn_output = self.masked_attn_multi(inputs, mask=True)

        # apply residual connection and normalize
        attn_output = self.attn_normalize(inputs[0] + attn_output)

        # apply feed-forward network to each position with relu activation
        ff_output = self.ff(attn_output)

        # apply dropout
        ff_output = self.ff_dropout(ff_output)

        # apply residual connection and normalize
        ff_output = self.ff_normalize(inputs[0] + ff_output)
        return ff_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'd_ff': self.d_ff,
                       'd_model': self.d_model,
                       'attn_heads': self.attn_heads,
                       'dropout_rate': self.dropout_rate})
        return config