import tensorflow as tf
#tf.config.set_visible_devices([],'GPU')
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.keras.utils.register_keras_serializable()
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  r=target-gen_output
  l2loss=tf.math.reduce_euclidean_norm(r)
  total_gen_loss = gan_loss +  100*l1_loss +l2loss

  return total_gen_loss, gan_loss, l1_loss

@tf.keras.utils.register_keras_serializable()
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss