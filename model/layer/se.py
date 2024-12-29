import tensorflow as tf


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, filters, ratio=4, name=None):
        super(SEBlock, self).__init__(name=name)
        self.gap = tf.keras.layers.GlobalAveragePooling1D()
        self.gmp = tf.keras.layers.GlobalMaxPooling1D()
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense1 = tf.keras.layers.Dense(
            int(filters // ratio),
            activation='gelu',
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        self.dense2 = tf.keras.layers.Dense(
            filters,
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        init = inputs
        se_mean = self.gap(inputs)
        se_max = self.gmp(inputs)
        se = self.concat([se_mean, se_max])
        se = self.dense1(se)
        se = self.dense2(se)
        x = self.multiply([init, se])

        return x


class TABlock(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(TABlock, self).__init__(name=name)
        
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense = tf.keras.layers.Conv1D(filters=1, kernel_size=1, activation='sigmoid')
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        init = inputs
        attention_weight_mean = tf.reduce_mean(inputs, axis=2, keepdims=True)
        attention_weight_max = tf.reduce_max(inputs, axis=2, keepdims=True)
        attention_weight = self.concat([attention_weight_mean, attention_weight_max])
        attention_weight = self.dense(attention_weight)
        x = self.multiply([init, attention_weight])

        return x