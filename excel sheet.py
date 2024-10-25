#Code to generate excel sheet for predicted conditions

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Path to the folder containing unlabelled images
test_images_dir = r'D:\Machine learning\Testing set\Testing set\Images'

# Path to your saved CNN model
cnn_model_save_path = r"D:\Machine learning\RESNET50 Models\6.SVM and CNN hybrid\cnn_model.h5"
# Path for saving predictions to Excel
output_excel_path = r'D:\Machine learning\RESNET50 Models\6.SVM and CNN hybrid\capsule_vision_predictions_format(for unpreprocessed images)(again).xlsx'

# Class names corresponding to the model's output classes
class_names = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']

# Define image size for ResNet50
image_size = (224, 224)

# Enable memory growth for GPU to avoid memory allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU.")
    except RuntimeError as e:
        print(e)

# Load the trained model
cnn_model = tf.keras.models.load_model(cnn_model_save_path)

# Function to load a single image
def load_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# List to store data rows for the DataFrame
data = []

# Iterate over all files in the test images directory
for img_name in tqdm(os.listdir(test_images_dir)):
    img_path = os.path.join(test_images_dir, img_name)
    
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image
        try:
            # Load the image without preprocessing
            img_array = load_image(img_path, image_size)
            
            # Predict the class probabilities for the image
            preds = cnn_model.predict(img_array)
            
            # Get the predicted class index and name
            pred_class_index = np.argmax(preds, axis=1)[0]
            pred_class_name = class_names[pred_class_index]
            
            # Prepare the data row
            # Row format: [image_name, probabilities for each class, predicted class]
            row = [img_name] + preds[0].tolist() + [pred_class_name]
            
            # Append row to the data list
            data.append(row)

        except Exception as e:
            print(f"Error processing image {img_name}: {e}")

# Create a DataFrame to store the predictions
df_predictions = pd.DataFrame(data, columns=['image_path'] + class_names + ['predicted_class'])

# Save predictions to Excel in the specified format
df_predictions.to_excel(output_excel_path, index=False)

print(f"Predictions saved to {output_excel_path}")
