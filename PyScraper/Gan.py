# SETTINGS
#batch_size = 128
#epochs = 15
#IMG_HEIGHT = 150
#IMG_WIDTH = 150
#IMG_DIR = './bing/avocadoes/'

#image_list = []

# Grab all of our images
#for filename in glob.glob(IMG_DIR + "*.jpg"):
#    im=Image.open(filename)
#    print(f"Loading {filename}")
#    image_list.append(im)

#print("Loaded " + str(len(image_list)) + " images")
import sys
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
from PIL import Image, ImageOps
import glob
import numpy as np
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# SETTINGS
IMG_DIR = './data/eyes/'

EPOCHS = 10000
GENERATE_RES = 3
noise_dim = 100
num_examples_to_generate = 16
load_model = True

image_list = []

BUFFER_SIZE = 700
BATCH_SIZE = 64
dropout_prob = 0.5

try: 
  # If already grabbed images, load it from the file
  print(f"Loading all data from {IMG_DIR}_data_npy.npy")
  image_list = np.load(IMG_DIR + "_data_npy.npy") 
except:
  # Grab all of our images
  imcount = 0;
  for filename in glob.glob(IMG_DIR + "*.jpg"):
    if imcount == 100:
      break
    #imcount += 1
    im=Image.open(filename)
    print(f"Loading {filename}")
    try:
      resize_image = im.resize((128, 128))
      gray_image = resize_image.convert("RGB")
      #image_list.append(gray_image)
      image_list.append(np.array(gray_image))
    except:
      print(f"Could not load {filename}")

  print("Loaded " + str(len(image_list)) + " images")
  
  np.save(IMG_DIR + "_data_npy.npy", np.asarray(image_list))

np_image_list = np.asarray(image_list)

train_images = np_image_list.reshape(np_image_list.shape[0], 128, 128, 3).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# ------- Display training images as test
#testimg = Image.new("RGB", (256, 256))

#i = 0
#print(train_images[0][0][0])
#for y in range(0, 4):
#  for x in range(0, 4):
#    img_np = (train_images[i])
#    img = Image.fromarray((((img_np + 1) / 2) * 255).astype(np.uint8))
#    testimg.paste(img, (y * 64, x * 64))
#    i += 1
#testimg.save("___test.png")
# -------------------

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(16*16*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))
    #assert model.output_shape == (None, 16, 16, 256) # Note: None is the batch size
    print(model.output_shape)

    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    #assert model.output_shape == (None, 16, 16, 128)
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    #assert model.output_shape == (None, 32, 32, 64)
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    #assert model.output_shape == (None, 64, 64, 3)
    print(model.output_shape)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    #assert model.output_shape == (None, 128, 128, 3)
    print(model.output_shape)

    if load_model:
      model.load_weights("C:/Users/kieth/source/repos/PyScraper/PyScraper/models/gen.h5")

    print("GENERATOR SUMMARY")
    model.summary()
    return model

def generator_loss(fake_output):
    retval = cross_entropy(tf.ones_like(fake_output), fake_output)
    print(f"{retval}")
    return retval

generator = make_generator_model()

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, 5, strides=2, padding='same', input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(256, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(512, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_prob))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))
    if load_model:
      model.load_weights("C:/Users/kieth/source/repos/PyScraper/PyScraper/models/disc.h5")
    print("DISCRIMINATOR SUMMARY")
    model.summary()
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    print(f"{total_loss}")
    return total_loss

discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def numpy2pil(array) -> Image:
  img = Image.fromarray(np.array(array), 'RGB')
  return img

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  result = Image.new("RGB", (512, 512))
  #print(predictions[0].numpy()[0][0])

  i = 0
  for y in range(0, 4):
    for x in range(0, 4):
      img_np = (predictions[i].numpy());
      img = Image.fromarray((((img_np + 1) / 2) * 255).astype(np.uint8))
      result.paste(img, (y * 128, x * 128))
      i += 1

  result.save('image_at_epoch_{:04d}.png'.format(epoch))

def train(dataset, epochs):
  totaltime = 0

  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator, epoch + 1, seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
      generator.save_weights("C:/Users/kieth/source/repos/PyScraper/PyScraper/models/gen.h5")
      discriminator.save_weights("C:/Users/kieth/source/repos/PyScraper/PyScraper/models/disc.h5")

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    totaltime += time.time()-start

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,epochs,seed)
  print ('Took {} seconds to train'.format(totaltime))
train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info > (6,2,0,''):
  display.Image(filename=anim_file)