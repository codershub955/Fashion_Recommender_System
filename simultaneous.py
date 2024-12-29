import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to extract skin regions using YCrCb color space
def extract_skin(image):
    # Convert to YCrCb color space for better skin tone detection
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Adjust the range for better skin tone capture
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    skin_mask = cv2.inRange(image_ycrcb, min_YCrCb, max_YCrCb)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin, skin_mask

# Function to find the dominant color in the skin region
def get_dominant_color(image, num_clusters=3):
    # Reshape image and remove zero values to focus on skin pixels
    image = image.reshape((-1, 3))
    image = image[~np.all(image == 0, axis=1)]

    # If no skin pixels are found, return a default color
    if len(image) == 0:
        return [128, 128, 128]  # Gray as a placeholder

    # KMeans to find the dominant color
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(image)

    # Find the largest cluster by counting labels
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant_color

# Load the image
image_path = "piyubuddha.png" # Path to the image file
image = cv2.imread(image_path)

# Extract skin regions using YCrCb
skin_ycrcb, skin_mask_ycrcb = extract_skin(image)

# Get dominant color in the YCrCb-detected skin region
skin_color_ycrcb = get_dominant_color(skin_ycrcb, num_clusters=3)
skin_color_ycrcb_rgb = tuple(map(int, skin_color_ycrcb))

print(f"Dominant Skin Color (RGB) from YCrCb + KMeans: {skin_color_ycrcb_rgb}")



# Second method: Using HSV color space and finding the average skin color
# Convert image to RGB (for displaying correctly)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to HSV color space for better color segmentation
hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# Define a refined skin color range in HSV for better accuracy
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Create a mask for skin color in HSV
skin_mask_hsv = cv2.inRange(hsv_image, lower_skin, upper_skin)

# Mask the original image to get the HSV skin region
skin_hsv = cv2.bitwise_and(image_rgb, image_rgb, mask=skin_mask_hsv)

# Find the average skin color in the HSV masked area
average_skin_color_hsv = cv2.mean(image_rgb, mask=skin_mask_hsv)[:3]

# Normalize the RGB value for display (divide by 255
#dnormalized_average_skin_color_hsv = np.array(average_skin_color_hsv) / 255.0

plt.subplot(1, 4, 1)
plt.axis('off')
plt.imshow([[normalized_average_skin_color_hsv]])  # Use normalized value
plt.title("Average Skin Color (HSV Mask)")

# Display results for both methods
plt.subplot(1, 4, 2)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(skin_mask_hsv, cmap='gray')
plt.title('Skin Mask (HSV)')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(skin_hsv)
plt.title('Detected Skin Area (HSV Mask)')
plt.axis('off')
plt.show()

print("Average Skin Color (RGB) from HSV Mask:",average_skin_color_hsv )
rgb = average_skin_color_hsv
rgb = (int(average_skin_color_hsv[0]), int(average_skin_color_hsv[1]), int(average_skin_color_hsv[2]))
hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
print("Hex color code : ",hex_color)