from models import Unet
from models import Discriminator
from utils import generator_loss
from utils import discriminator_loss
from time import perf_counter
from layers import downsample
import tensorflow as tf 
import gc

import os
import numpy as np
import datetime
from datapipeline import load_random_labeled_pair

@tf.keras.utils.register_keras_serializable()
def jaccard_index(y_true, y_pred):
    intersection = np.logical_and(y_pred,y_true).sum()
    union = np.logical_or(y_true, y_pred).sum()
    ji=intersection / union if union!=0 else 0
    return ji
@tf.keras.utils.register_keras_serializable()
def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_index(y_true, y_pred)
@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection) / (union + 1e-7)  # Adding epsilon to avoid division by zero
@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)



unet=Unet()
discriminator=Discriminator()
Unet_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-6, beta_1=0.5)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(Unet_optimizer=Unet_optimizer,Unet=unet)

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Checkpoint restored from", latest_checkpoint)
else:
    print("No checkpoint found. Starting from scratch.")

log_dir="logs/"
for i in range(12):
    unet.layers[i].trainable = False



summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape,tf.GradientTape() as l1tape,tf.GradientTape() as Ttape:
    gen_output,_ = unet(input_image, training=True)
    disc_real_output = discriminator([input_image, target], training=True)
    input_image_reshaped = tf.reshape(input_image, (1, 256, 256, 3))

# Reshape gen_output to match the discriminator's expected input shape
    gen_output_reshaped = tf.reshape(gen_output, (1, 256, 256, 1))
    disc_generated_output = discriminator([input_image_reshaped, gen_output_reshaped], training=True)
    target = tf.cast(target, tf.float32)
    gen_output = tf.cast(gen_output, tf.float32)
    disc_real_output=tf.cast(disc_real_output, tf.float32)
    disc_generated_output=tf.cast(disc_generated_output, tf.float32)
    jl=jaccard_loss(gen_output,target)
    di=dice_coefficient(gen_output,target)
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  Unet_gradients = gen_tape.gradient(gen_gan_loss,unet.trainable_variables)
  Unetl1_gradients = l1tape.gradient(gen_l1_loss,unet.trainable_variables)
  #UnetT_gradients = Ttape.gradient(jl,unet.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

  Unet_optimizer.apply_gradients(zip(Unet_gradients,unet.trainable_variables))
  Unet_optimizer.apply_gradients(zip(Unetl1_gradients,unet.trainable_variables))
  #Unet_optimizer.apply_gradients(zip(UnetT_gradients,unet.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('dice', di, step=step)
    tf.summary.scalar('jaccard', jl, step=step)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
    tf.summary.scalar('disc_loss', disc_loss, step=step)



def train_loop(num_epochs, num_train_samples, num_val_samples):
    best_loss = float('inf')
    best_epoch = 0
    vl=[]
    t0=perf_counter()
    nv=2
    for epoch in range(num_epochs):
        gen_total_lossv=0
        gen_gan_lossv=0
        gen_l1_lossv=0
        disc_lossv=0
        jlv=0
        div=0
        for _ in range(num_train_samples):
          input_data,target_data = load_random_labeled_pair()
          train_step(input_data[tf.newaxis, ...],target_data[tf.newaxis, ...],epoch*num_train_samples+_)
        for _ in range(num_val_samples):
          input_data,target_data = load_random_labeled_pair()
          gen_output,_ = unet(input_data[tf.newaxis, ...])
          disc_real_output = discriminator([input_data[tf.newaxis, ...],target_data[tf.newaxis, ...]])
          disc_generated_output = discriminator([input_data[tf.newaxis, ...],target_data[tf.newaxis, ...]])
          target_data = tf.cast(target_data, tf.float32)
          gen_output = tf.cast(gen_output, tf.float32)
          disc_real_output=tf.cast(disc_real_output, tf.float32)
          disc_generated_output=tf.cast(disc_generated_output, tf.float32)
          gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target_data[tf.newaxis, ...])
          disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
          jl=jaccard_loss(gen_output,target_data)
          di=dice_coefficient(gen_output,target_data)
          gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target_data)
          disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
          gen_l1_lossv+=gen_l1_loss
          disc_lossv+=disc_loss
          jlv+=jl
          div+=di

        val_loss =[gen_l1_lossv/num_val_samples,disc_lossv,div,jlv]
        gc.collect()
        vl.append(val_loss)

        if(epoch%20==0):
            a=np.array(vl)
            np.savetxt("er",a)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss}")


        # Save the best model
        if gen_l1_lossv < best_loss :
            best_loss = gen_l1_lossv
            best_epoch = epoch
            checkpoint.save(file_prefix=checkpoint_prefix)

    print(f"Best model saved at epoch {best_epoch + 1}, with validation loss: {best_loss}")

#train_loop(10000,32,0)