# Puts scraped data into .npy file
import glob
import numpy as np
from PIL import Image

# SETTINGS
DATA_FOLDER = 'eyes'
IMG_DIR = f'./data/{DATA_FOLDER}/'
imcount = 0
max_images = 100
image_list = []

image_size = 128

# Grab all of our images
for filename in glob.glob(IMG_DIR + "*.jpg"):
  if imcount == max_images:
    break
  #imcount += 1
  im=Image.open(filename)
  print(f"Loading {filename}")
  try:
    resize_image = im.resize((image_size, image_size), Image.ANTIALIAS)
    gray_image = resize_image.convert("RGB")
    image_list.append(np.array(gray_image))
  except:
    print(f"Could not load {filename}")

print("Loaded " + str(len(image_list)) + " images")
image_list = np.reshape(np.asarray(image_list), (-1, image_size, image_size, 3))
image_list = image_list / 127.5-1

print()
print()
print(f"Saved data as _{DATA_FOLDER}.npy")
np.save(f"_{DATA_FOLDER}.npy", image_list)
