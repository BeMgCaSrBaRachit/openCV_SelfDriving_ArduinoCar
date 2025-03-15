import cv2
import numpy as np

# Load the image
image = cv2.imread("maze.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Define RGB color thresholds
colors = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}

# Convert image to numpy array
height, width, _ = image.shape
output = np.zeros_like(image)

# Function to find closest color
def closest_color(pixel):
    distances = {color: np.linalg.norm(np.array(pixel) - np.array(rgb)) for color, rgb in colors.items()}
    return colors[min(distances, key=distances.get)]

# Process each pixel
for i in range(height):
    for j in range(width):
        output[i, j] = closest_color(image[i, j])

# Save and display the output
output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
cv2.imwrite("output.jpg", output_bgr)
cv2.imshow("Digital RGB Image", output_bgr)
cv2.waitKey(0)
cv2.destroy