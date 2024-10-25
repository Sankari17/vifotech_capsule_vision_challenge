#Preprocessing the Augmented datasets

import os
import numpy as np
from PIL import Image, ImageOps
import cv2

# Function to apply preprocessing steps and save images while keeping the folder structure
def preprocess_and_save_images(input_dir, output_dir, size=(224, 224)):
    # Walk through the input directory tree
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check if the file is an image based on its extension
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                input_image_path = os.path.join(root, file)

                # Create the equivalent output directory structure
                relative_path = os.path.relpath(root, input_dir)
                output_dir_with_structure = os.path.join(output_dir, relative_path)
                os.makedirs(output_dir_with_structure, exist_ok=True)

                # Generate output image path
                output_image_path = os.path.join(output_dir_with_structure, file)

                try:
                    with Image.open(input_image_path) as img:
                        # 1. Resize the image
                        img = img.resize(size)

                        # 2. Convert to grayscale (optional)
                        # img = img.convert('L')

                        # 3. Normalize (scaling pixel values to [0, 1])
                        img_array = np.array(img) / 255.0  # For [0, 1] normalization

                        # 4. Histogram equalization (optional)
                        # img_array = cv2.equalizeHist(img_array.astype(np.uint8))

                        # 5. Convert back to PIL image
                        img = Image.fromarray((img_array * 255).astype(np.uint8))

                        # 6. Optional: Perform image rotation or flip (data augmentation)
                        # img = img.rotate(90)  # Example rotation
                        # img = ImageOps.flip(img)  # Example flip

                        # Save the preprocessed image in the corresponding output folder
                        img.save(output_image_path)

                except Exception as e:
                    print(f"Error processing {input_image_path}: {e}")

# File paths for training and validation sets
training_input_dir = r"D:\Machine learning\Extracted files\Dataset\Dataset\Training balanced"
validation_input_dir = r"D:\Machine learning\Extracted files\Dataset\Dataset\Validation balanced"

# Output paths to save preprocessed images (maintain folder structure)
training_output_dir = r"D:\Machine learning\On whole preprocessed dataset(Augmented)\Whole preprocessed train dataset(augmented)"
validation_output_dir = r"D:\Machine learning\On whole preprocessed dataset(Augmented)\Whole preprocessed validation dataset(augmented)"

# Preprocess training set images and preserve folder structure
preprocess_and_save_images(training_input_dir, training_output_dir, size=(224, 224))

# Preprocess validation set images and preserve folder structure
preprocess_and_save_images(validation_input_dir, validation_output_dir, size=(224, 224))

print("Preprocessing complete for both training and validation sets, with folder structure preserved.")
