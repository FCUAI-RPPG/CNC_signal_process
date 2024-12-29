import tensorflow as tf

from .transformer import positional_encoding

class FeatureFFN(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.project = tf.keras.layers.Dense(d_model, use_bias=False)
        self.seq1 = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.seq2 = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()
        self.norm0 = tf.keras.layers.BatchNormalization()
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.norm2 = tf.keras.layers.BatchNormalization()
        

    def call(self, x):
        x = self.project(x)
        x = self.norm0(x)

        x = self.add1([x, self.seq1(x)])
        x = self.norm1(x)

        x = self.add2([x, self.seq2(x)])
        x = self.norm2(x)
        return x


class SEFusion(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.project1 = tf.keras.layers.Dense(d_model, use_bias=False)
        self.project2 = tf.keras.layers.Dense(d_model, use_bias=False)
        self.concat = tf.keras.layers.Concatenate()
        self.norm = tf.keras.layers.BatchNormalization()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model*2, activation='sigmoid'),
        ])
        self.multiply = tf.keras.layers.Multiply()
        self.downsample = tf.keras.layers.Dense(d_model)
        

    def call(self, spectrogram, feature):
        spectrogram = self.project1(spectrogram)
        feature = self.project2(feature)
        x = self.concat([spectrogram, feature])
        x = self.norm(x)
        att_weight = self.seq(x)
        x = self.multiply([x, att_weight])
        x = self.downsample(x)
        return x


class FIBlock(tf.keras.layers.Layer):
    # Frquency Interaction Block
    def __init__(self, filters, name=None):
        super(FIBlock, self).__init__(name=name)
        self.time_project = tf.keras.layers.Dense(filters)
        self.frquency_project = tf.keras.layers.Dense(filters)
        self.fuse_project = tf.keras.layers.Dense(filters)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.add = tf.keras.layers.Add()
        
    def call(self, time_x, frquency_x):
        time_x = self.time_project(time_x)
        frquency_x = self.frquency_project(frquency_x)
        fuse_x = self.concat([time_x, frquency_x])
        fuse_x = self.fuse_project(fuse_x)
        fuse_x = self.add([time_x, fuse_x])
        return fuse_x


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, name=None):
        super().__init__(name=name)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, workpiece_idx):
        workpiece_idx = tf.gather(self.pos_encoding, workpiece_idx)
        return workpiece_idx


class PositionalScaling(tf.keras.layers.Layer):
    def __init__(self, d_model, name=None):
        super().__init__(name=name)
        self.d_model = d_model

    def call(self, x):
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return x


class FrequencyTransform(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.permute1 = tf.keras.layers.Permute((1,3,2))
        self.permute2 = tf.keras.layers.Permute((1,3,2))

    def call(self, x):
        x = self.permute1(x)
        x = tf.abs(tf.signal.rfft(x))
        x = self.permute2(x)
        return x