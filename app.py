import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import requests

# Define the URL for the model file
model_url = 'https://drive.google.com/file/d/18X6mSr2NaCZGnGsZcyHqURs4LHd1cFEQ/view?usp=share_link'

# Download the model file
response = requests.get(model_url)
with open('model_016950.h5', 'wb') as f:
    f.write(response.content)

# Load your trained model
model = load_model('model_016950.h5')

# Load and preprocess your custom image
def load_and_preprocess_image(image_path, target_shape=(256, 256)):
    # Open the image using PIL
    custom_image = Image.open(image_path)
    
    # Resize the image to the desired shape
    custom_image = custom_image.resize(target_shape)
    
    # Convert the image to an array
    custom_image = np.array(custom_image)
    
    # Assuming the input image is in RGB format
    # Check if the image needs normalization (pixel values in [0, 255] range)
    if custom_image.max() > 1.0:
        # Normalize the pixel values to the range [-1, 1]
        custom_image = (custom_image.astype(np.float32) - 127.5) / 127.50
    
    # Add a batch dimension
    custom_image = np.expand_dims(custom_image, axis=0)
    
    return custom_image


# Plot source and generated images
def plot_images(src_img, gen_img):
    images = np.vstack((src_img, gen_img))
    # Scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    titles = ['Original', 'Generated']
    # Plot images row by row
    for i in range(len(images)):
        # Define subplot
        plt.subplot(1, 2, 1 + i)
        # Turn off axis
        plt.axis('off')
        # Plot raw pixel data
        plt.imshow(images[i])
        # Show title
        plt.title(titles[i])
    st.pyplot()

# Streamlit app
st.title("GAN Image Generator")

# Upload custom image
custom_image = st.file_uploader("Upload a custom image", type=["png", "jpg", "jpeg"])

# Check if an image is uploaded
if custom_image:
    # Generate an image from your custom source
    gen_image = model.predict(load_and_preprocess_image(custom_image))

    # Plot both images together
    plot_images(load_and_preprocess_image(custom_image), gen_image)
