from layers import downsample
from layers import upsample
import tensorflow as tf
from layers import con
import tensorflow as tf

def magclassifier():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(16, 4, False)(x)  # (batch_size, 128, 128, 16)
    down2 = downsample(32, 4)(down1)     # (batch_size, 64, 64, 32)
    down3 = downsample(64, 4)(down2)     # (batch_size, 32, 32, 64)
    n1=tf.keras.layers.BatchNormalization()(down3)
    conv_last = tf.keras.layers.Conv2D(60, 1, strides=1,
                                        kernel_initializer=initializer)(n1)
    flat=tf.keras.layers.Flatten()(conv_last)
    den1=tf.keras.layers.Dense(256)(flat)
    n2=tf.keras.layers.BatchNormalization()(den1)
    den2=tf.keras.layers.Dense(128)(n2)
    den3=tf.keras.layers.Dense(120)(den2)


    # Apply softmax activation to get one-hot encoding
    output = tf.keras.layers.Softmax(axis=-1)(den3)

    return tf.keras.Model(inputs=inp, outputs=output)


import tensorflow as tf

def shuffclassifierenc(input_shape=(None,2048)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Add hidden layers
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    # Output layer with softmax activation
    outputs = tf.keras.layers.Dense(24, activation='softmax')(x)
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def shuffclassifierencpos(input_shape=(None, 2048)):
    input_1 = tf.keras.layers.Input(shape=input_shape)
    input_2 = tf.keras.layers.Input(shape=input_shape)
    
    # Concatenate the inputs along the last axis
    concatenated_inputs = tf.keras.layers.Concatenate(axis=-1)([input_1, input_2])
   
    # Add hidden layers
    x = tf.keras.layers.Dense(1024, activation='relu')(concatenated_inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # Output layer with softmax activation
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    
    # Define the model
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=outputs)
    
    return model



def shuffclassifier():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')

    x = inp

    down1 = downsample(16, 4, False)(x)  # (batch_size, 128, 128, 16)
    down2 = downsample(32, 4)(down1)     # (batch_size, 64, 64, 32)
    down3 = downsample(64, 4)(down2)     # (batch_size, 32, 32, 64)

    conv_last = tf.keras.layers.Conv2D(1, 1, strides=1,
                                        kernel_initializer=initializer)(down3)
    flat=tf.keras.layers.Flatten()(conv_last)
    den1=tf.keras.layers.Dense(512)(flat)
    den2=tf.keras.layers.Dense(256)(den1)
    den3=tf.keras.layers.Dense(24)(den2)


    # Apply softmax activation to get one-hot encoding
    output = tf.keras.layers.Softmax(axis=-1)(den3)

    return tf.keras.Model(inputs=inp, outputs=output)


def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')


  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 64, 64, channels*2)

  down1 = downsample(16, 4, False)(x)  # (batch_size, 128, 128, 8)
  down2 = downsample(32, 4)(down1)  # (batch_size, 64, 64, 8)
  batchnorm0 = tf.keras.layers.BatchNormalization()(down2)
  down3 = downsample(64, 4)(batchnorm0)  # (batch_size, 32, 32, 8)
 
  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 10, 10, 8)
  conv = tf.keras.layers.Conv2D(8, 3, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 8, 8, 8)
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 10, 10, 8)

  last = tf.keras.layers.Conv2D(1, 1, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 10, 10, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)



def Unet():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(32, 3, apply_batchnorm=False),  # (batch_size, 128, 128, 16)
        downsample(64, 3),  # (batch_size, 64, 64, 32)
        downsample(64, 3),  # (batch_size, 32, 32, 32)
        downsample(128, 3),  # (batch_size, 16, 16, 64)
        downsample(128, 3),  # (batch_size, 8, 8, 64)
        downsample(256, 3),  # (batch_size, 4, 4, 128)
        downsample(512, 3),  # (batch_size, 2, 2, 256)
    ]

    up_stack = [
        con(),
        upsample(256, 4, apply_dropout=True),  # (batch_size, 4, 4, 128)
        upsample(128, 4, apply_dropout=True),  # (batch_size, 8, 8, 64)
        upsample(128, 4),  # (batch_size, 16, 16, 64)
        upsample(64, 4),  # (batch_size, 32, 32, 32)
        upsample(64, 4),  # (batch_size, 64, 64, 32)
        upsample(32, 4),  # (batch_size, 128, 128, 16)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='sigmoid')  # (batch_size, 256, 256, 1)

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips[::-1]):  # Reversed order of skips
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        
    x=tf.keras.layers.BatchNormalization()(x)
    x = last(x) 
     # Final output
    
    # Output of the encoder part as a vector
    encoder_output = tf.keras.layers.Flatten()(skips[-1])

    return tf.keras.Model(inputs=inputs, outputs=[x, encoder_output])



