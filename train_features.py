from models import Unet
from time import perf_counter
import tensorflow as tf 
import gc
import numpy as np
import os
import numpy as np
import keras
import datetime
from datapipeline import enhanced
unet=Unet()
Unet_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(Unet_optimizer=Unet_optimizer,Unet=unet)
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Checkpoint restored from", latest_checkpoint)
else:
    print("No checkpoint found. Starting from scratch.")

log_dir="logsfet/"
def L2(y_true,y_pred):
    r=y_true-y_pred
    l2loss=tf.math.reduce_euclidean_norm(r)
    return l2loss
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
los=keras.losses.CategoricalFocalCrossentropy()
def train_step(input_image,piece, step):
  with tf.GradientTape() as disc_tape,tf.GradientTape() as Ttape:
    _,ogencoded = unet(input_image, training=True)
    _,encoded = unet(piece, training=True)
    cl=L2(ogencoded,encoded)

  UnetT_gradients = Ttape.gradient(cl,unet.trainable_variables)

  Unet_optimizer.apply_gradients(zip(UnetT_gradients,unet.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('los', cl, step=step)


def train_loop(num_epochs, num_train_samples, num_val_samples):
    best_loss = float('inf')
    best_epoch = 0
    vl=[]
    t0=perf_counter()
    for epoch in range(num_epochs):
        clv=0
        for _ in range(num_train_samples):
          input,piece=enhanced()
          train_step(input[np.newaxis, ...],piece[np.newaxis, ...],epoch*num_train_samples+_)
        for _ in range(num_val_samples):
            input_image,inputpiece=enhanced()
            _,encoded = unet(inputpiece[np.newaxis, ...], training=True)
            _,ogencoded = unet(input_image[np.newaxis, ...], training=True)
            cl=L2(encoded,ogencoded)
            clv+=cl
        clv/=num_val_samples
        vl.append([clv,perf_counter()-t0])
        if(clv<best_loss):
            a=np.array(vl)
            print(clv)
            checkpoint.save(checkpoint_prefix)
            best_loss=clv
            best_epoch=epoch
            np.savetxt("er",a)
        if(epoch%30==0):
           a=np.array(vl)
           np.savetxt("erfe",a)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {clv}")
        with summary_writer.as_default():
          tf.summary.scalar('val_los', clv, step=epoch)

    print(f"Best model saved at epoch {best_epoch + 1}, with validation loss: {best_loss}")

#train_loop(10000,32,8)