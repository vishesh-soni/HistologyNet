

folder_path="/home/obasho/Documents/hackethon/data/unlabelled/images"
images_dir="/home/obasho/Documents/hackethon/data/labelled/images"
masks_dir="/home/obasho/Documents/hackethon/data/labelled/masks"
import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

def one_hot_encode_list(input_list):
    """
    One-hot encodes a list of integers representing unique values from 0 to 5.

    Args:
    - input_list (list of int): Input list of integers.

    Returns:
    - one_hot_encoded_vector (numpy.ndarray): One-hot encoded vector.
    """

    # Define the possible values for each element
    values = list(range(4))  # Values from 0 to 5
    num_values = len(values)
    # Generate all possible permutations
    permutations = [(x, y, z, l) for x in values for y in values if y != x for z in values if z != x and z != y for l in values if l!=x and l!=y and l!=z]
    # Create a mapping between permutation and index
    permutation_to_index = {permutation: i for i, permutation in enumerate(permutations)}

    # Get the index of the input permutation
    index = permutation_to_index[tuple(input_list)]

    # Create a one-hot encoded vector for the input permutation
    one_hot_encoded_vector = np.zeros(len(permutations))
    one_hot_encoded_vector[index] = 1
    one_hot_encoded_vector = np.reshape(one_hot_encoded_vector, (-1, len(permutations)))

    return one_hot_encoded_vector


def positionpice():
    image_files = os.listdir(folder_path)

    # Select a random image file
    random_image_file = random.choice(image_files)

    # Construct the full path to the image file
    image_path = os.path.join(folder_path, random_image_file)

    # Read the original image
    original_image = np.array(Image.open(image_path))

    height, width, _ = original_image.shape
    patch_size = (width // 4, height // 4)
    magnification_position = (random.randint(0, width - patch_size[0]), random.randint(0, height - patch_size[1]))

    # Magnify the selected patch 16 times
    magnified_patch = original_image[magnification_position[1]:magnification_position[1] + patch_size[1],
                                     magnification_position[0]:magnification_position[0] + patch_size[0], :]
    magnified_image = np.array(Image.fromarray(magnified_patch).resize((patch_size[0] * 4, patch_size[1] * 4)))

    # Determine the quadrant of the magnified patch
    quadrant = (magnification_position[0] >= width // 2) * 2 + (magnification_position[1] >= height // 2)
    quadrant_one_hot = tf.one_hot(quadrant, depth=4)
    quadrant_one_hot = np.reshape(quadrant_one_hot, (-1, 4))

    return original_image,magnified_image, quadrant_one_hot



def cut_and_shuffle_image(image, n_pieces):
    """
    Cuts the image into n_pieces and shuffles them randomly.

    Args:
    - image (numpy.ndarray): Input image (RGB).
    - n_pieces (int): Number of pieces to cut the image into.

    Returns:
    - shuffled_image (numpy.ndarray): Shuffled image.
    """
    # Get the dimensions of the image
    height, width, channels = image.shape

    # Calculate the size of each piece
    piece_height = height // n_pieces
    piece_width = width // n_pieces

    # Initialize list to store pieces
    pieces = []
    piece_positions = []


    # Cut the image into pieces
    for i in range(n_pieces):
        for j in range(n_pieces):
            piece = image[i*piece_height:(i+1)*piece_height, j*piece_width:(j+1)*piece_width, :]
            pieces.append(piece)
            piece_positions.append(i*n_pieces+j)

    # Shuffle the pieces randomly
    random_seed = 42
    random.Random(random_seed).shuffle(pieces)
    random.Random(random_seed).shuffle(piece_positions)

    # Reshape the shuffled pieces to (n_pieces, n_pieces, piece_height, piece_width, channels)
    reshaped_pieces = np.array(pieces).reshape(n_pieces, n_pieces, piece_height, piece_width, channels)

    # Initialize list to store rows of pieces
    rows = []

    # Concatenate pieces in each row
    for i in range(n_pieces):
        row = np.concatenate(reshaped_pieces[i], axis=1)
        rows.append(row)

    # Concatenate rows to form the shuffled image
    shuffled_image = np.concatenate(rows, axis=0)
    shuffled_image = np.pad(shuffled_image, 
                                     ((0, 256 - shuffled_image.shape[0]), 
                                      (0, 256 - shuffled_image.shape[1]), 
                                      (0, 0)), 
                                     mode='constant')
    return shuffled_image,piece_positions


def read_shuffled_image_unlabeled(n_pieces):
    """
    Read a random image from the unlabeled folder, cut it into n pieces, shuffle them randomly,
    and return both the shuffled original and grayscale images.

    Args:
    - folder_path (str): Path to the unlabeled folder containing images.
    - n_pieces (int): Number of pieces to cut the image into.

    Returns:
    - shuffled_original_image (numpy.ndarray): Shuffled original RGB image.
    - grayscale_image (numpy.ndarray): Grayscale image.
    """
    # List all files in the folder
    image_files = os.listdir(folder_path)

    # Select a random image file
    random_image_file = random.choice(image_files)

    # Construct the full path to the image file
    image_path = os.path.join(folder_path, random_image_file)

    # Read the original image using TensorFlow
    original_image = tf.io.read_file(image_path)
    original_image = tf.image.decode_image(original_image, channels=3)  # Decode the image (RGB)

    # Convert the original image to numpy array
    original_image = original_image.numpy()

    # Cut and shuffle the original image
    shuffled_original_image,pos = cut_and_shuffle_image(original_image, n_pieces)
    label=one_hot_encode_list(pos)
    # Convert the original image to grayscale
    grayscale_image = tf.image.rgb_to_grayscale(original_image)

    # Convert the grayscale image to numpy array
    grayscale_image = grayscale_image.numpy()

    return shuffled_original_image, grayscale_image,label





def enhance_contrast_with_original(image):
    """
    Enhance the contrast of an image by scaling its intensity values.

    Parameters:
        image (tf.Tensor): Input image tensor.
        contrast_factor (float): Contrast enhancement factor.
                                 - Values greater than 1 increase contrast.
                                 - Values between 0 and 1 decrease contrast.

    Returns:
        tuple: A tuple containing the original and contrast-enhanced image tensors.
    """
    ix=np.random.rand()
    contrast_factor=ix*20+0.4
    # Convert image to floating-point format for calculations
    image_float = tf.cast(image, tf.float32)

    # Normalize image to range [0, 1]
    image_normalized = image_float / 255.0

    # Apply contrast enhancement using linear transformation
    enhanced_image = (image_normalized - 0.5) * contrast_factor + 0.5

    # Clip pixel values to ensure they remain in range [0, 1]
    enhanced_image = tf.clip_by_value(enhanced_image, 0.0, 1.0)

    # Convert back to uint8 format
    enhanced_image = tf.cast(enhanced_image * 255.0, tf.uint8)

    return image, enhanced_image




def gaussian_kernel(size, sigma):
    """
    Generate a 2D Gaussian kernel.

    Parameters:
        size (int): Size of the kernel (width and height).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        tf.Tensor: 2D Gaussian kernel tensor.
    """
    ax = tf.range(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / tf.reduce_sum(kernel)

def edge_detection(image):
    """
    Perform edge detection on the input image using the Canny edge detector.

    Parameters:
        image (tf.Tensor): Input image tensor.

    Returns:
        tuple: A tuple containing the original image tensor and the resized edge-detected image tensor.
    """
    # Convert image to grayscale
    grayscale_image = tf.image.rgb_to_grayscale(image)

    # Convert image to float32 format for compatibility with convolution operation
    grayscale_image_float = tf.cast(grayscale_image, tf.float32)

    # Perform Gaussian blur to reduce noise
    kernel_size = 5
    sigma = 1.4
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    blurred_image = tf.nn.conv2d(tf.expand_dims(grayscale_image_float, axis=0), kernel, strides=[1, 1, 1, 1], padding='SAME')

    # Perform Canny edge detection
    edges = tf.image.sobel_edges(blurred_image)
    edge_magnitude = tf.norm(edges, axis=-1)
    edge_detection_threshold = 50  # Adjust threshold as needed
    edge_mask = tf.cast(edge_magnitude > edge_detection_threshold, tf.float32)

    # Resize edge-detected image tensor to match the size of the original image
    resized_edges = tf.squeeze(edge_mask)  # Remove the extra dimension

    return image, resized_edges

def image_contrast(image):
    """
    Compute the contrast of an image using TensorFlow.

    Parameters:
        image (tf.Tensor): Input image tensor.

    Returns:
        tf.Tensor: Contrast of the input image.
    """
    # Convert image to grayscale
    grayscale_image = tf.image.rgb_to_grayscale(image)

    # Cast grayscale image to float32
    grayscale_image_float32 = tf.cast(grayscale_image, tf.float32)

    # Compute mean pixel intensity
    mean_intensity = tf.reduce_mean(grayscale_image_float32)

    # Compute squared difference of pixel intensities from the mean
    squared_diff = tf.square(grayscale_image_float32 - mean_intensity)

    # Compute variance (average of squared differences)
    variance = tf.reduce_mean(squared_diff)

    # Compute contrast as the square root of variance
    contrast = tf.sqrt(variance)

    return contrast

getfe=random.choice([edge_detection,enhance_contrast_with_original])
def enhanced():
    image_files = os.listdir(folder_path)

    # Select a random image file
    random_image_file = random.choice(image_files)

    # Construct the full path to the image file
    image_path = os.path.join(folder_path, random_image_file)

    # Read the original image using TensorFlow
    original_image = tf.io.read_file(image_path)
    original_image = tf.image.decode_image(original_image, channels=3)
    original_image,new=getfe(original_image)
    return original_image,new

def read_random_image_unlabeled():
    """
    Read a random image from the unlabeled folder, convert it to grayscale,
    and return both the original and grayscale images.

    Args:
    - folder_path (str): Path to the unlabeled folder containing images.

    Returns:
    - original_image (numpy.ndarray): Original RGB image.
    - grayscale_image (numpy.ndarray): Grayscale image.
    """
    # List all files in the folder
    image_files = os.listdir(folder_path)

    # Select a random image file
    random_image_file = random.choice(image_files)

    # Construct the full path to the image file
    image_path = os.path.join(folder_path, random_image_file)

    # Read the image using TensorFlow
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)  # Decode the image (RGB)

    # Convert the image to grayscale
    grayscale_image = tf.image.rgb_to_grayscale(image)

    # Convert images to numpy arrays
    original_image = image.numpy()
    grayscale_image = grayscale_image.numpy()

    return original_image, grayscale_image

def load_random_labeled_pair():
    # List all image files in the directories
    image_files = os.listdir(images_dir)
    mask_files = os.listdir(masks_dir)
    
    # Select a random file
    random_file = random.choice(image_files)
    
    # Load the selected image and mask
    image_path = os.path.join(images_dir, random_file)
    mask_path = os.path.join(masks_dir, random_file)
    
    # Read image and mask using TensorFlow
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3)  # Ensure 3 channels for image
    
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_image(mask, channels=1)  # Read mask as single channel
    
    return image, mask
