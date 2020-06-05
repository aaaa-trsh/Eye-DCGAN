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

IMG_DIR = './data/eyes/'

image_list = []
image_gray_list = []
save_images = True
# Grab all of our images
imcount = 0;
for filename in glob.glob(IMG_DIR + "*.jpg"):
  if imcount == 16:
    break
  imcount+=1
  im=Image.open(filename)
  print(f"Loading {filename}")
  resize_image = im.resize((64, 64))
  #ImageOps.invert(resize_image)
  gray_image = resize_image.convert("RGB")
  image_list.append(gray_image)
  image_gray_list.append(np.array(gray_image))

print("Loaded " + str(len(image_list)) + " images")

np_image_list = np.asarray(image_gray_list)

result = Image.new("RGB", (256, 256))

i = 0
for y in range(0, 4):
  for x in range(0, 4):
    img = (np_image_list[i])
    result.paste(Image.fromarray(img), (y * 64, x * 64))
    i += 1
result.save("___test.png")