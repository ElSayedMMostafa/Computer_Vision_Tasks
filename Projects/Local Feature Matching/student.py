import numpy as np
from scipy.signal import convolve2d
from skimage.feature import peak_local_max
import cv2

def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    # Harris Corner Detector
    # Apply Harris detector (Manually)
    R = apply_harris(image, ksize=3, k=0.04, w_size=4)
    # Define a threshold and apply binary thresholding on R values
    thresh = 0.02*R.max()
    ret, threshold_image = cv2.threshold(R, thresh, 1.0, 0, cv2.THRESH_BINARY)
    # Non-maximum suppression (min_disatnce is a hyperparameter)
    p_image = peak_local_max(threshold_image, min_distance= 20)

    xs = p_image[:, 1] #Corners X-Coordinates
    ys = p_image[:, 0] #Corners Y-Coordinates

    return xs, ys

def get_features(image, x, y, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """
    # TODO: Your implementation here! See block comments and the project webpage for instructions

    num_points = x.shape[0]
    features = np.zeros((num_points, 128))
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    filtered_image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0.8, sigmaY=0.8)
    # Compute the gradient
    Ix = convolve2d(filtered_image, sobel_x, 'same') # (4*4 cell)
    Iy = convolve2d(filtered_image, sobel_y, 'same')  # (4*4 cell)
    # Compute the magnitudes and orientations
    magnitudes = np.sqrt(np.square(Ix) + np.square(Iy))
    orientations = np.arctan2(Iy, Ix)
    orientations[orientations < 0] += 2 * np.pi
    L = int(feature_width / 2)
    point_index = -1
    # For each point of interest, get the histogram
    # (1) Cut a patch of (feature_width * feature_width) around the interest point
    for x_point, y_point in zip(x, y):
        # Work for an interest point ::
        point_index += 1
        # Get the patch (feature_width * feature_width)
        x_point, y_point = int(x_point), int(y_point)
        patch = filtered_image[y_point-L:y_point+L, x_point-L:x_point+L]
        # check for correct dimensions
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            continue
        # Get the corresponding gradient magnitudes and orientations
        patch_mag = magnitudes[y_point-L:y_point+L, x_point-L:x_point+L]
        patch_ori = orientations[y_point-L:y_point+L, x_point-L:x_point+L]
        # Get the histogram for each patch/cells
        #features[point_index, :] = patch_to_features(patch, feature_width)
        histogram = []
        # (1) Cut each patch into cells abd get the corresponding histogram for each cell and hence for each patch
        for iy in range(0, patch.shape[0], 4):
            for ix in range(0, patch.shape[1], 4):
                #curr_cell = np.array(patch[iy:iy + 4, ix:ix + 4])  # (4*4 cell)
                curr_mag = np.array(patch_mag[iy:iy + 4, ix:ix + 4]).flatten()
                curr_ori = np.array(patch_ori[iy:iy + 4, ix:ix + 4]).flatten()
                #cell_histogram = np.histogram(curr_ori, bins=8, range=(0, 2*np.pi), weights=curr_mag)[0]
                cell_histogram = get_histogram(curr_ori, curr_mag, n_pins=8, range_=(0, 2 * np.pi))[0]
                # Concatenating the histograms for one patch histogram
                histogram.extend(cell_histogram)

        histogram = np.array(histogram).reshape(1, -1)
        #Normalize the histogram
        histogram_norm = np.linalg.norm(histogram)
        #histogram_norm[histogram_norm == 0] = 1 #to avoid division by zero [nan value]
        histogram /= histogram_norm
        # Add the histogram to the features vector
        features[point_index, :] = histogram
    #print('features_shape = ', features.shape)
    return features

def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    my_matches = []
    my_confidences = []
    matching_threshold = 0.9
    # Loop for each feature in image 1
    for feat1_index in range(im1_features.shape[0]):
        feat1 = im1_features[feat1_index]
        # Get the euclidean distance with each feature in image 2
        distances = np.array([np.linalg.norm(feat1-feat2) for feat2 in im2_features])
        # Sort the distances ascendingly
        distances_indicies = np.argsort(distances)
        # Get the distances of the 2 nearest features
        d1 = distances[distances_indicies[0]]
        d2 = distances[distances_indicies[1]]
        # Use NN ratio test
        if d1/d2 < matching_threshold:
            # Good match
            my_matches.append([feat1_index, distances_indicies[0]])
            my_confidences.append(1- d1/d2)

    my_matches = np.array(my_matches)
    my_confidences = np.array(my_confidences)
    return my_matches, my_confidences

def get_histogram(orientations, weights, n_pins = 8, range_ = (0, 2*np.pi)):
    min_val, max_val = range_
    assert max_val > min_val
    #hist_values = np.linspace(min_val, max_val,num= n_pins)
    histo = np.zeros((1, n_pins))
    cmp_val = (max_val-min_val)/n_pins
    for i in range(orientations.shape[0]):
        index = min(int(orientations[i]/cmp_val), n_pins-1)
        histo[0, index] += weights[i]
    return histo

def apply_harris(img, ksize=3, k=0.04, w_size = 4):
    """
    Applies Harris corner detection algorithm on an image

    :params:
    :img: a grayscale or color image
    :ksize: kernel size of Gaussien filter
    :k: Hyperparameter betwen detH and TraceH**2
    :w_size: Window size around each point

    :returns:
    :R: a np array that holds all values after applying Harris corner detection (same size as the image)
    """
    # Assert Even Window Size
    assert w_size % 2 == 0
    rows, cols = img.shape
    # Smoothing
    img = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=1, sigmaY=1)
    # Sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # Compute the 1st derivatives with Gaussien weights
    Ix = convolve2d(img, sobel_x, 'same')
    Iy = convolve2d(img, sobel_y, 'same')
    Ix = cv2.GaussianBlur(Ix, ksize=(ksize, ksize), sigmaX=0.8, sigmaY=0.8)
    Iy = cv2.GaussianBlur(Iy, ksize=(ksize, ksize), sigmaX=0.8, sigmaY=0.8)
    # Compute the 2nd derivatives
    Ixx = np.square(Ix)
    Iyy = np.square(Iy)
    Ixy = np.multiply(Ix, Iy)

    # Summations Over Windows
    Sxx = np.zeros((rows, cols))
    Syy = np.zeros((rows, cols))
    Sxy = np.zeros((rows, cols))
    R = np.zeros((rows, cols))

    s_size = int(w_size/2)
    for row in range(s_size, rows-w_size-1):
        for col in range(s_size, cols-w_size-1):
            Sxx[row, col] = np.sum(Ixx[row-s_size:row+s_size, col-s_size:col+s_size])
            Syy[row, col] = np.sum(Iyy[row-s_size:row+s_size, col-s_size:col+s_size])
            Sxy[row, col] = np.sum(Ixy[row-s_size:row+s_size, col-s_size:col+s_size])
            det = Sxx[row, col]*Syy[row, col] - Sxy[row, col]**2
            trace = Sxx[row, col]+Syy[row, col]
            R[row, col] = det - k*(trace**2)
    return R

def get_features_patch_normalized(image, x, y, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """
    num_points = x.shape[0]
    features = np.zeros((num_points, 256))
    #return features
    (x, y) = (y, x)  # replacing
    # TODO: Your implementation here! See block comments and the project webpage for instructions
    L = int(feature_width / 2)
    point_index = -1
    for x_point, y_point in zip(x, y):
        print(x_point, y_point)
        print(image.shape)
        point_index += 1
        # Get the patch (feature_width * feature_width)
        x_point, y_point = int(x_point), int(y_point)
        patch = image[x_point-L:x_point+L, y_point-L:y_point+L]
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            continue
        # Get the histogram for each patch/cells
        #features[point_index, :] = patch_to_features(patch, feature_width)
        histogram = []
        for ix in range(0, patch.shape[0], 4):
            for iy in range(0, patch.shape[1], 4):
                curr_cell = np.array(patch[ix:ix + 4, iy:iy + 4]).reshape(1,16)[0]
                histogram.extend(curr_cell)
        histogram = np.array(histogram).reshape(1,-1)
        histogram /= np.linalg.norm(histogram)
        features[point_index, :] = histogram

    return features

