
import numpy as np
import cv2
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale


def create_gaussian_filter(ksize, sigma=(1, 1)):
    '''''
    - A functions creates a gaussian filter of an arbitary size (ksize*ksize)
    @Parameters: 
        ksize: odd integer that represents the length and width of the mean filter
        sigma: a tuble/list of 2 elements represents (x_sigma, y_sigma), where sigma is the standard deviation of each direction
               However, the default value if 1 for both.
    @Returns:
        gaussian_filter: The obtained mean filter
    '''''
    assert ksize % 2 != 0

    # For the value of A, in case of sigma_x != sigma_y, I'll use the mean of both sigmas to evaluate the value of A
    A = 1 / (2 * np.pi * np.mean(sigma))
    # Creating a linspace for each direction (2D)
    x = np.linspace(start=0, stop=ksize - 1, num=ksize)
    y = x.copy()
    # Calculate the mean of each direction (One value as they're the same linspace)
    mu = np.mean(x)
    # Create the meshgrid that represents the 2D values as a grid
    X, Y = np.meshgrid(x, y)
    # Applying the gaussian filter equation on the created grid
    gaussian_filter = A * np.exp((-(X - mu) ** 2) / (2 * sigma[0]) + (-(Y - mu) ** 2) / (2 * sigma[1]))
    return gaussian_filter

def my_zero_padding(X, x_levels=1, y_levels=1):
    if X.ndim > 2:
        X = X[:,:,0]

    padded = np.zeros((X.shape[0]+2*x_levels, X.shape[1]+2*y_levels))
    padded[x_levels:X.shape[0]+x_levels, y_levels:X.shape[1]+y_levels] = X
    return padded

def reflective_padding(X, x_levels=1, y_levels=1):
    if X.ndim > 2:
        X = X[:,:,0]

    x_rows, x_cols = X.shape[0], X.shape[1]
    # Fill with zeros
    padded = my_zero_padding(X, x_levels, y_levels)
    #
    # [1] rows -- Save the edges
    levels_above = np.flip(X[:x_levels, :], 0)
    levels_below = np.flip(X[x_rows - x_levels:x_rows, :], 0)
    # Add the edges
    padded[0:x_levels, y_levels:x_cols + y_levels] = levels_above
    padded[x_rows + x_levels:, y_levels:x_cols + y_levels] = levels_below

    # [2] cols -- Save the edges
    levels_left = np.flip(padded[:, y_levels:2 * y_levels], 1)
    levels_right = np.flip(padded[:, x_cols:x_cols + y_levels], 1)
    # Add the edges
    padded[:, 0:y_levels] = levels_left
    padded[:, x_cols + y_levels:] = levels_right
    return padded

def conv_2d(x, k):
    # x: The image
    # k: The Kernel
    k_xlength, k_ylength = k.shape[0], k.shape[1]
    assert k_xlength % 2 == 1
    assert k_ylength % 2 == 1

    x_levels = k_xlength // 2  # the x levels to be increased (zero padded)
    y_levels = k_ylength // 2  # the y levels to be increased (zero padded)

    image_x, image_y = x.shape[0], x.shape[1]
    ## Perform Zero Padding
    padded_image = my_zero_padding(x, x_levels, y_levels)
    ## Convolution
    conv_output = np.zeros(x.shape)
    for i in range(image_x):
        for j in range(image_y):
            sub_img = padded_image[i:i + k_xlength, j:j + k_ylength]
            conv_output[i, j] = np.sum(np.multiply(sub_img, k))
    return conv_output

def my_imfilter(image: np.ndarray, filter: np.ndarray):

    k_x_length, k_y_length = filter.shape[0], filter.shape[1]
    assert k_x_length % 2 == 1
    assert k_y_length % 2 == 1

    if image.ndim == 2:
        filtered_image = conv_2d(image, filter)
    else:
        output0 = conv_2d(image[:, :, 0], filter)
        output1 = conv_2d(image[:, :, 1], filter)
        output2 = conv_2d(image[:, :, 2], filter)
        filtered_image = np.dstack((output0, output1, output2))
    return filtered_image

def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float):
  """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

   Task:
   - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
   - Combine them to create 'hybrid_image'.
  """

  assert image1.shape == image2.shape
  kernel_size = 3 * cutoff_frequency
  gaussian_kernel = create_gaussian_filter(kernel_size, sigma=(cutoff_frequency, cutoff_frequency))
  low_frequencies = my_imfilter(image1, gaussian_kernel)
  high_frequencies = image2 - my_imfilter(image2, gaussian_kernel)
  hybrid_image = low_frequencies + high_frequencies
  return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image: np.ndarray):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect', multichannel=True)
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))
