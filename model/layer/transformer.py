import numpy as np
import tensorflow as tf


def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, x.dtype))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.add = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(rate=0.1)


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        attn_output = self.dropout(attn_output)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='gelu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


# class EncoderLayer(tf.keras.layers.Layer):
#     def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.):
#         super().__init__()

#         self.self_attention = GlobalSelfAttention(
#             num_heads=num_heads,
#             key_dim=d_model,
#             dropout=dropout_rate)

#         self.ffn = FeedForward(d_model, dff)

#     def call(self, x):
#         x = self.self_attention(x)
#         x = self.ffn(x)
#         return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        # (batch_size, target_len, target_vocab_size)
        logits = self.final_layer(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits


class FeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self, *, d_model):
        super(FeatureEmbedding, self).__init__()
        self.d_model = d_model

        self.feature_project = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-12)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.feature_project(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        x = self.layernorm(x)
        return x


class FeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = FeatureEmbedding(d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embedding(x)

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x


class FusionEmbedding(tf.keras.layers.Layer):
    def __init__(self, *, d_model):
        super(FusionEmbedding, self).__init__()
        self.d_model = d_model

        self.spectrogram_embedding = tf.keras.layers.Dense(d_model)
        self.mm_embedding = tf.keras.layers.Embedding(3, d_model)
        self.cut_embedding = tf.keras.layers.Embedding(2, d_model)
        self.rmp_embedding = tf.keras.layers.Dense(d_model)
        self.fr_embedding = tf.keras.layers.Dense(d_model)
        self.add = tf.keras.layers.Add()
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-12)

    def call(self, spectrogram, mm, cut, rpm, feed_rate):
        length = tf.shape(spectrogram)[1]

        spectrogram = self.spectrogram_embedding(spectrogram)
        mm = self.mm_embedding(mm)
        cut = self.cut_embedding(cut)
        rpm = self.mm_embedding(rpm)
        feed_rate = self.fr_embedding(feed_rate)
        
        x = self.add([spectrogram, mm, cut, rpm, feed_rate])
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        x = self.layernorm(x)

        return x


class FusionDecoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 dropout_rate=0.1):
        super(FusionDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = FusionEmbedding(d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

    def call(self, spectrogram, mm, cut, rpm, feed_rate, features):
        x = self.pos_embedding(spectrogram, mm, cut, rpm, feed_rate)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, features)

        return x[:,-1,:]


def EncoderLayer(d_model, num_heads, dff, lora, name):
    def forward(inputs):
        layer_inputs = inputs

        att_outputs = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            dropout=0,
        )(layer_inputs)

        layer_inputs = tf.keras.layers.Dense(dff)(att_outputs)
        if lora:
            delta_h_down = tf.keras.layers.Dense(units=d_model//4, kernel_initializer='glorot_normal', name=f"{name}_LoRA_ffn1_down")(att_outputs)
            delta_h_up = tf.keras.layers.Dense(units=dff, kernel_initializer='glorot_normal', name=f"{name}_LoRA_ffn1_up")(delta_h_down)
            layer_inputs = tf.keras.layers.Add(name=f"{name}_LoRA_add1")([layer_inputs, delta_h_up])
        act = tf.keras.layers.Activation(activation='gelu')(layer_inputs)
        layer_inputs = tf.keras.layers.Dense(d_model)(act)
        if lora:
            delta_h_down = tf.keras.layers.Dense(units=d_model//4, kernel_initializer='glorot_normal', name=f"{name}_LoRA_ffn2_down")(act)
            delta_h_up = tf.keras.layers.Dense(units=d_model, kernel_initializer='glorot_normal', name=f"{name}_LoRA_ffn2_up")(delta_h_down)
            layer_inputs = tf.keras.layers.Add(name=f"{name}_LoRA_add2")([layer_inputs, delta_h_up])
        layer_inputs = tf.keras.layers.Dropout(rate=0.1)(layer_inputs)
        layer_inputs = tf.keras.layers.Add()([att_outputs, layer_inputs])
        outputs = tf.keras.layers.LayerNormalization(epsilon=1e-12)(layer_inputs)

        return outputs

    return forward


def ConvFeatureExtractionModel(filters, conv_layers):
    def forward(inputs):
        layer_inputs = inputs

        for i, cl in enumerate(conv_layers):
            (dim, k, stride) = cl
            layer_inputs = tf.keras.layers.Conv1D(
                filters=dim,
                kernel_size=k,
                strides=stride,
                use_bias=False,
                kernel_initializer='he_normal',
            )(layer_inputs)
            layer_inputs = tf.keras.layers.Dropout(rate=0.1)(layer_inputs)
            if i == 0:
                layer_inputs = tf.keras.layers.LayerNormalization(epsilon=1e-12)(layer_inputs)
            layer_inputs = tf.keras.layers.Activation(activation='gelu')(layer_inputs)
        
        layer_inputs = tf.keras.layers.LayerNormalization(epsilon=1e-12)(layer_inputs)
        layer_inputs = tf.keras.layers.Dense(units=filters)(layer_inputs)
        layer_inputs = PositionalEmbedding(d_model=filters)(layer_inputs)
        layer_inputs = tf.keras.layers.Dropout(rate=0.1)(layer_inputs)
        layer_inputs = tf.keras.layers.LayerNormalization(epsilon=1e-12)(layer_inputs)
        return layer_inputs

    return forward