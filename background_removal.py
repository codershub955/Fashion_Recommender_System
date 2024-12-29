import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'skin.png'  # Path to the image file
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to HSV color space for better color segmentation
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Define a skin color range in HSV
# These values can be adjusted based on testing with different skin tones
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Create a mask to detect skin regions
skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

# Apply the mask to the original image to isolate the skin region
skin = cv2.bitwise_and(image, image, mask=skin_mask)

# Calculate the average skin color in the masked area
average_skin_color = cv2.mean(image, mask=skin_mask)[:3]

# Display the original image, skin mask, and extracted skin color
plt.figure(figsize=(12, 4))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Skin Mask
plt.subplot(1, 3, 2)
plt.imshow(skin_mask, cmap='gray')
plt.title('Skin Mask')
plt.axis('off')

# Extracted skin area
plt.subplot(1, 3, 3)
plt.imshow(skin)
plt.title('Detected Skin Area')
plt.axis('off')

# Show the plots
plt.show()

# Output the average skin color
print("Average Skin Color (RGB):", average_skin_color)
