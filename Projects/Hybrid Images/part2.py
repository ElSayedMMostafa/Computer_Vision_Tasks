
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from helpers import vis_hybrid_image, load_image, save_image, my_imfilter, gen_hybrid_image
from skimage.transform import resize

## Setup
# Read images and convert to floating point format
image1 = load_image('../data/einstein.bmp')
image2 = load_image('../data/marilyn.bmp')
image2 = resize(image2, image1.shape)
# display the dog and cat images
plt.figure(figsize=(3,3))
plt.imshow((image1*255).astype(np.uint8))
plt.figure(figsize=(3,3))
plt.imshow((image2*255).astype(np.uint8))


## Hybrid Image Construction ##
cutoff_frequency = 7
low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(image1, image2, cutoff_frequency)

high_frequencies = np.clip(high_frequencies + 0.5, a_min=0.0, a_max=1.0)
hybrid_image = np.clip(hybrid_image, a_min=0.0, a_max=1.0)
## Visualize and save outputs ##
plt.figure()
plt.imshow((low_frequencies*255).astype(np.uint8))
plt.figure()
plt.imshow(((high_frequencies+0.5)*255).astype(np.uint8))
vis = vis_hybrid_image(hybrid_image)
plt.figure(figsize=(20, 20))
plt.imshow(vis)

save_image('../results/low_frequencies.jpg', low_frequencies)
save_image('../results/high_frequencies.jpg', high_frequencies)
save_image('../results/hybrid_image.jpg', hybrid_image)
save_image('../results/hybrid_image_scales.jpg', vis)
