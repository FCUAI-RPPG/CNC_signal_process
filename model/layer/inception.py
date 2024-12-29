import tensorflow as tf

from .se import SEBlock, TABlock


def inception(filters, kernel_size, lora, name):
    def forward(inputs):
        # BranchA
        conv_a = tf.keras.layers.Dense(units=filters, name=f"{name}_conv_a")(inputs)

        # BranchB
        down_b = tf.keras.layers.Dense(units=filters, name=f"{name}_down_b")(inputs)
        conv_b = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1, dilation_rate=1, padding="same", name=f"{name}_conv_b")(down_b)

        # BranchC
        down_c = tf.keras.layers.Dense(units=filters, name=f"{name}_down_c")(inputs)
        conv_c = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1, dilation_rate=2, padding="same", name=f"{name}_conv_c")(down_c)

        # BranchD
        pool_d = tf.keras.layers.MaxPool1D(pool_size=kernel_size, strides=1, padding="same", name=f"{name}_pool_d")(inputs)
        conv_d = tf.keras.layers.Dense(units=filters, name=f"{name}_conv_d")(pool_d)

        # Concate
        cat = tf.keras.layers.Concatenate(name=f"{name}_concate", axis=-1)([conv_a, conv_b, conv_c, conv_d])
        dropout = tf.keras.layers.Dropout(rate=0.1, name=f"{name}_dropout")(cat)
        ln = tf.keras.layers.LayerNormalization(epsilon=1e-12, name=f"{name}_norm")(dropout)
        ffn = tf.keras.layers.Dense(units=filters*4, name=f"{name}_ffn")(ln)

        if lora:
            delta_h_down = tf.keras.layers.Dense(units=filters//2, kernel_initializer='glorot_normal', name=f"{name}_LoRA_ffn_down")(ln)
            delta_h_up = tf.keras.layers.Dense(units=filters*4, kernel_initializer='glorot_normal', name=f"{name}_LoRA_ffn_up")(delta_h_down)
            ffn = tf.keras.layers.Add(name=f"{name}_LoRA_add")([ffn, delta_h_up])
        
        act = tf.keras.layers.Activation(activation="gelu", name=f"{name}_act")(ffn)
        ta = TABlock(name=f"{name}_ta")(act)
        se = SEBlock(filters=filters*4, name=f"{name}_se")(ta)
        
        # Residual
        if inputs.shape[-1] != se.shape[-1]:
            residual = tf.keras.layers.Dense(units=filters*4, name=f"{name}_residual_conv")(inputs)
            residual = tf.keras.layers.Activation(activation="gelu", name=f"{name}_residual_act")(residual)
        else:
            residual = inputs
        outputs = tf.keras.layers.Add(name=f"{name}_add_residual")([residual, se])

        return outputs

    return forward


def stem(filters, conv_layers):
    def forward(inputs):
        layer_inputs = inputs

        layer_inputs = tf.keras.layers.Dense(units=filters, name="stem_input_linear")(layer_inputs)

        for i, cl in enumerate(conv_layers):
            (k, stride) = cl
            layer_inputs = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=k,
                strides=stride,
                name=f'stem_conv_{i}',
            )(layer_inputs)
        layer_inputs = tf.keras.layers.Dropout(rate=0.1, name=f"stem_output_dropout")(layer_inputs)
        layer_inputs = tf.keras.layers.LayerNormalization(epsilon=1e-12, name=f"stem_output_norm")(layer_inputs)
        layer_inputs = tf.keras.layers.Dense(units=filters*4, name="stem_output_linear")(layer_inputs)
        layer_inputs = tf.keras.layers.Activation(
            activation='gelu',
            name=f'stem_output_act'
        )(layer_inputs)
        return layer_inputs

    return forward