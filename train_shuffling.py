from models import Unet
from models import shuffclassifierenc
from time import perf_counter
import tensorflow as tf 
import gc
import os
import numpy as np
import datetime
from datapipeline import read_shuffled_image_unlabeled
#tf.config.set_visible_devices([],'GPU')
@tf.keras.utils.register_keras_serializable()
def SsimLoss(y_true, y_pred):
    y_pred_uint8 = tf.cast(y_pred * 255, tf.uint8)
    y_true_uint8 = tf.cast(y_true * 255, tf.uint8)
    l = tf.image.ssim(y_pred_uint8, y_true_uint8, max_val=1.0)
    lo=(1-l)*100
    return lo

unet=Unet()
clasif=shuffclassifierenc()
Unet_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(Unet_optimizer=Unet_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 Unet=unet,
                                 shuffclassifier=unet)

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Checkpoint restored from", latest_checkpoint)
else:
    print("No checkpoint found. Starting from scratch.")

log_dir="logsshff/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def train_step(input_image, onehot,target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape,tf.GradientTape() as l1tape,tf.GradientTape() as Ttape:
    gen_output,encoded = unet(input_image, training=True)
    encoded_with_batch = tf.expand_dims(encoded, axis=0)
    labelout = clasif(encoded_with_batch, training=True)  
    labelout = tf.squeeze(labelout, axis=0) 
    cl=tf.keras.losses.categorical_crossentropy(labelout,onehot)

  #Unet_gradients = gen_tape.gradient(sim,unet.trainable_variables)
  #Unetl1_gradients = l1tape.gradient(gen_total_loss,unet.trainable_variables)
  UnetT_gradients = Ttape.gradient(cl,unet.trainable_variables)
  discriminator_gradients = disc_tape.gradient(cl,clasif.trainable_variables)


  #Unet_optimizer.apply_gradients(zip(Unet_gradients,unet.trainable_variables))
  #Unet_optimizer.apply_gradients(zip(Unetl1_gradients,unet.trainable_variables))
  Unet_optimizer.apply_gradients(zip(UnetT_gradients,
                                          unet.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              clasif.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('los', cl[0], step=step)



def train_loop(num_epochs, num_train_samples, num_val_samples):
    best_loss = float('inf')
    best_epoch = 0
    vl=[]
    t0=perf_counter()
    for epoch in range(num_epochs):
        clv=0
        simv=0
        for _ in range(num_train_samples):
          input_data,target_data,label = read_shuffled_image_unlabeled(2)
          train_step(input_data[tf.newaxis, ...],label,target_data[tf.newaxis, ...],epoch*num_train_samples+_)
        for _ in range(num_val_samples):
            input_data,target_data,label = read_shuffled_image_unlabeled(2)
            _,encoded = unet(input_data[tf.newaxis, ...], training=True)
            encoded_with_batch = tf.expand_dims(encoded, axis=0)
            labelout=clasif(encoded_with_batch, training=True)
            labelout = tf.squeeze(labelout, axis=0) 
            cl=tf.keras.losses.categorical_crossentropy(labelout,label)
            clv+=(cl[0]/num_val_samples)
        gc.collect()
        vl.append([clv,perf_counter()-t0])
        if(epoch%30==0):
           a=np.array(vl)
           np.savetxt("er",a)
        if(simv<best_loss):
            a=np.array(vl)
            print(clv)
            checkpoint.save(checkpoint_prefix)
            best_loss=simv
            best_epoch=epoch
            np.savetxt("ersh",a)
        #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {clv}")
        with summary_writer.as_default():
          tf.summary.scalar('val_los', clv, step=epoch)

    print(f"Best model saved at epoch {best_epoch + 1}, with validation loss: {best_loss}")

#train_loop(10000,32,4)