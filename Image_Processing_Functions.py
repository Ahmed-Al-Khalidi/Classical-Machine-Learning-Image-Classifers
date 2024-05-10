################################################################################################
import os
import cv2
import numpy as np
################################################################################################
from PIL import Image
import matplotlib.pyplot as plt
################################################################################################
import seaborn as sns
from skimage.feature import hog, corner_harris, corner_peaks, local_binary_pattern
from skimage.color import rgb2gray
from skimage.filters import sobel
################################################################################################

def image_pipeline_with_resize(folder_path,height,width):
    images_data = []
    labels = []
    size_=(width,height)
    Classes = os.listdir(folder_path)

    for i in range(len(Classes)):
        new_path = folder_path + '/' + Classes[i]
        Classes2 = os.listdir(new_path)

        for j in range(len(Classes2)):
            

            image_path = folder_path + '/' + Classes[i] + '/' + Classes2[j]
            
            try:
                # Attempt to open the image
                image = Image.open(image_path)

                # Resize the images
                img_resized = image.resize(size_, Image.Resampling.LANCZOS)
                
                # Convert the image to a NumPy array
                image_array = np.array(img_resized)
                
                # Append the image array to the list
                images_data.append(image_array)
                
                labels.append(Classes[i])
            
            except (IOError, OSError) as e:
                # Handle the exception (e.g., print a warning, skip the image, etc.)
                print(f"Error processing {image_path}: {e}")

    return images_data, labels


def RGB_to_grayscale(image_array):
    length= len(image_array)
    images_list=[]
    for i in range(length):
        # Compute the weighted sum to convert to grayscale
        images_list.append(np.dot(image_array[i], [0.299, 0.587, 0.114]))

    return images_list


def Image_scalling(data_list):
    list_length=len(data_list)
    new_data_list=[]
    for i in range(list_length):
        new_data_list.append((data_list[i]/255))

    return new_data_list


# Function to extract HOG features
def HOG_extract_features(images):
    features = []
    for image in images:
        hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
        features.append(hog_features)
    return np.array(features)



def Image_extract_features(images, method='hog'):
    features = []
    for image in images:
        if method == 'hog':
            hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
            features.append(hog_features)
        elif method == 'corner_harris':
            gray_image = rgb2gray(image)
            corners = corner_peaks(corner_harris(gray_image), min_distance=5)
            features.append(corners)
        elif method == 'local_binary_pattern':
            gray_image = rgb2gray(image)
            lbp = local_binary_pattern(gray_image, P=8, R=1)
            features.append(lbp.ravel())  # ravel to convert 2D array to 1D
        elif method == 'sobel':
            sobel_image = sobel(rgb2gray(image))
            features.append(sobel_image.ravel())  # ravel to convert 2D array to 1D
        else:
            raise ValueError("Invalid method. Supported methods are 'hog', 'corner_harris', 'local_binary_pattern', and 'sobel'.")
    return np.array(features)






