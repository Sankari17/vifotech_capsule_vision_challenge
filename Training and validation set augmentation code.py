#Augmentation of Training and Validation set

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Directories (replace with your paths)
train_dir = r'D:\Machine learning\Extracted files\Dataset\Dataset\Training'
validation_dir = r'D:\Machine learning\Extracted files\Dataset\Dataset\Validation'

# Output directories for augmented images
train_augmented_dir = r'D:\Machine learning\Extracted files\Dataset\Dataset\Training balanced'
validation_augmented_dir = r'D:\Machine learning\Extracted files\Dataset\Dataset\Validation balanced'

# Define augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to get all image paths from a class directory
def get_all_image_paths(class_dir):
    image_paths = []
    for root, dirs, files in os.walk(class_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                image_paths.append(os.path.join(root, file))
    return image_paths

# Function to augment a specific class and save in corresponding output directory
def augment_class(class_name, source_dir, target_dir, augment_count):
    class_dir = os.path.join(source_dir, class_name)
    augmented_dir = os.path.join(target_dir, class_name)

    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)

    # Get all image paths for the current class
    current_images = get_all_image_paths(class_dir)
    current_count = len(current_images)

    print(f"Class: {class_name}, Current Images Count: {current_count}")  # Debugging line

    # Check if there are any images in the current class
    if current_count == 0:
        print(f"Warning: No images found in class '{class_name}' to augment.")
        return

    # Augment images to match the required count
    for i in range(augment_count):
        image_path = current_images[i % current_count]
        print(f"Augmenting: {image_path}")  # Debugging line
        image = np.expand_dims(plt.imread(image_path), axis=0)
        
        # Augment image and save in the target directory
        for batch in datagen.flow(image, batch_size=1, save_to_dir=augmented_dir, save_prefix=class_name, save_format='jpg'):
            break  # Only one image per augmentation

# Function to balance the dataset by augmenting classes
def balance_dataset(class_distribution, source_dir, target_dir):
    normal_count = class_distribution['Normal']

    for class_name, count in class_distribution.items():
        if class_name != 'Normal':
            augment_count = normal_count - count
            if augment_count > 0:
                augment_class(class_name, source_dir, target_dir, augment_count)
                print(f'Augmented {augment_count} images for class {class_name}')

# Get class distribution for training and validation sets
training_class_distribution = {
    'Angioectasia': 1154, 'Bleeding': 834, 'Erosion': 2694, 'Erythema': 691,
    'Foreign Body': 792, 'Lymphangiectasia': 796, 'Normal': 28663, 
    'Polyp': 1162, 'Ulcer': 663, 'Worms': 158
}

validation_class_distribution = {
    'Angioectasia': 497, 'Bleeding': 359, 'Erosion': 1155, 'Erythema': 297,
    'Foreign Body': 340, 'Lymphangiectasia': 343, 'Normal': 12287, 
    'Polyp': 500, 'Ulcer': 286, 'Worms': 68
}

# Balance training dataset and save augmented images in the training directory
balance_dataset(training_class_distribution, train_dir, train_augmented_dir)

# Balance validation dataset and save augmented images in the validation directory
balance_dataset(validation_class_distribution, validation_dir, validation_augmented_dir)
