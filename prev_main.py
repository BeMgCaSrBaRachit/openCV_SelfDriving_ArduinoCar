import cv2
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import sys
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

img_original = plt.imread("maze.jpeg")
cv_image = cv2.imread("maze.jpeg")
cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

img = img_original
print('Program Running......')
vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]

n,m,d = img.shape
edges = np.zeros((n,m))
for row in range(3,n-2):
    for col in range(3,m-2):
        local_pixels = img[row-1 : row+2, col-1: col+2, 0] #This 0 is here because my filteration matrix is 3D
        vertical_transformed_pixels = vertical_filter*local_pixels
        vertical_score = vertical_transformed_pixels.sum()/4

        horizontal_transformed_pixels = horizontal_filter*local_pixels
        horizontal_score = horizontal_transformed_pixels.sum()/4

        edge_score = math.sqrt(math.pow(horizontal_score,2) + math.pow(vertical_score,2))
        edges[row][col] = 3*edge_score
minX = sys.maxsize
minY = sys.maxsize

maxX  = -1
maxY = -1
maxVal = -1
for row in range(5,n-3): #constant X
    curr = edges[row][int(m/2)]
    next = edges[row+1][int(m/2)]
    if(curr - next >= 150):
        minY = min(minY,row)
    if(next - curr >= 150):
        maxY = max(maxY, row)

for col in range(5,m-3): #constant Y
    curr = edges[int(n/2)][col]
    next = edges[int(n/2)][col+1]
    if(next - curr >= 100):
        minX = min(minX,col)
    if(curr - next >= 100):
        maxX = max(maxX, col)

crop_image = img_original[minY:maxY, minX:maxX]
original_crop_image = crop_image
pure_red = (255, 0, 0)
pure_green = (0, 255, 0)
white = (255, 255, 255)

# Define HSV ranges for red and green
color_ranges = {
    "red1": [(0, 120, 50), (10, 255, 255)],  # Lower red range
    "red2": [(170, 120, 50), (180, 255, 255)],  # Upper red range
    "green": [(40, 50, 50), (90, 255, 255)],  # Green range
}

def extract_red_green(image):
    image = image.copy()  # Ensure the image is writable
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    output = np.full_like(image, white)  # Start with all white pixels

    # Detect red and green
    mask_red1 = cv2.inRange(hsv, np.array(color_ranges["red1"][0]), np.array(color_ranges["red1"][1]))
    mask_red2 = cv2.inRange(hsv, np.array(color_ranges["red2"][0]), np.array(color_ranges["red2"][1]))
    mask_green = cv2.inRange(hsv, np.array(color_ranges["green"][0]), np.array(color_ranges["green"][1]))

    # Combine both red masks
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Assign colors
    output[mask_red > 0] = pure_red
    output[mask_green > 0] = pure_green

    return output, mask_red, mask_green

# Load input image
image = crop_image  # Assuming `crop_image` is defined

# Process the image
output, mask_red, mask_green = extract_red_green(image)

# Find contours for red and green regions
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Function to get bounding box
def get_bounding_box(contours):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        return (x, y, w, h)  # Return bounding box as (x, y, width, height)
    return None

# Get bounding boxes
red_box = get_bounding_box(contours_red)
green_box = get_bounding_box(contours_green)

# Draw bounding boxes
output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
rotation = 0
# Determine relative position of the green box w.r.t. the red box
if red_box and green_box:
    rx, ry, rw, rh = red_box  # Red box (x, y, width, height)
    gx, gy, gw, gh = green_box  # Green box (x, y, width, height)

    # Midpoints of the bounding boxes
    red_mid_x, red_mid_y = rx + rw // 2, ry + rh // 2
    green_mid_x, green_mid_y = gx + gw // 2, gy + gh // 2

    # Compare midpoints to determine relative position
    if green_mid_x < red_mid_x:
        position = "Left"
        rotation = 90
    elif green_mid_x > red_mid_x:
        position = "Right"
        rotation = -90
    elif green_mid_y < red_mid_y:
        position = "Forward"
        rotation = 0
    else:
        position = "Backward"
        rotation = 180
    print(f"The green box is {position} relative to the red box.")

# Convert back to RGB for displaying
output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

# Display output
# plt.imshow(output_rgb)
# plt.axis("off")
# plt.show()

def rotate(image,angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)  # Keep the rotation around the center
    # Compute rotation matrix (keeps axes fixed)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle * -1, 1.0)
    # Keep the image size the same after rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return rotated_image

def convert_to_primary_colors(image):
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, blue, white, and black
    color_ranges = {
        "red1": [(0, 50, 50), (10, 255, 255)],  # Lower red
        "red2": [(170, 50, 50), (180, 255, 255)],  # Upper red
        "blue": [(90, 50, 50), (130, 255, 255)],  # Blue range
        "white": [(0, 0, 200), (180, 50, 255)],  # White range
        "black": [(0, 0, 0), (180, 255, 50)],  # Black range
    }

    # Create an output image (start as white)
    output = np.full_like(image, (255, 255, 255))  # Default to white

    # Detect and replace colors
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        if color == "red1" or color == "red2":
            output[mask > 0] = (0, 0, 255)  # Pure Red (BGR)
        elif color == "blue":
            output[mask > 0] = (255, 0, 0)  # Pure Blue (BGR)
        elif color == "white":
            output[mask > 0] = (255, 255, 255)  # Pure White
        elif color == "black":
            output[mask > 0] = (0, 0, 0)  # Pure Black

    return output

scaled_crop_image = cv2.resize(original_crop_image, (5,5), cv2.INTER_LINEAR_EXACT)
scaled_crop_image = rotate(scaled_crop_image, rotation)
scaled_crop_image = convert_to_primary_colors(scaled_crop_image)

final_matrix = np.ones((5,5))
source = (0,0)
destination = (0,0)
for rows in range(0,5):
    for col in range(0,5):
        r = round(scaled_crop_image[rows][col][0]/255)
        g = round(scaled_crop_image[rows][col][1]/255)
        b = round(scaled_crop_image[rows][col][2]/255)
        # print(r,g,b)
        if (r == g == b):
            if r == 1:
                final_matrix[rows][col] = 1
            else:
                final_matrix[rows][col] = 0
        else:
            if r == 1:
                source = (col, rows)
            elif b == 1:
                destination = (col, rows)
Matrix = np.array(final_matrix)
# print(Matrix)
print(source, destination)

grid = Grid(matrix=Matrix)
start = grid.node(source[0], source[1])
end = grid.node(destination[0], destination[1])

finder = AStarFinder(diagonal_movement=0)

path,runs = finder.find_path(start, end, grid)
pathString = ""
currDirection = "t"
prev = 0
print(path)
for node in path:
    if(prev == 0):
        prev = node
    if prev!=node:
        x = node.x - prev.x
        y = node.y - prev.y
        if(x == 1 and y == 0): #absolute right
            if(currDirection == "t"):
                pathString+="rf"
                currDirection = "r"
            elif(currDirection == "r"):
                pathString+="f"
            elif(currDirection == "b"):
                pathString+="lf"
                currDirection = "r"
            elif(currDirection == "l"):
                pathString+="rrf"
                currDirection = "r"
        if(x == -1 and y == 0): #absolute left
            if(currDirection == "t"):
                pathString+="lf"
                currDirection = "l"
            elif(currDirection == "r"):
                pathString+="llf"
                currDirection = "l"
            elif(currDirection == "b"):
                pathString+="rf"
                currDirection = "l"
            elif(currDirection == "l"):
                pathString+="f"
        if(x == 0 and y == -1): #absolute top
            if(currDirection == "t"):
                pathString+="f"
                currDirection = "t"
            elif(currDirection == "r"):
                pathString+="lf"
                currDirection = "t"
            elif(currDirection == "b"):
                pathString+="rrf"
                currDirection = "t"
            elif(currDirection == "l"):
                pathString+="rf"
                currDirection = "t"
        if(x == 0 and y == 1): #absolute bottom
            if(currDirection == "t"):
                pathString+="llf"
                currDirection = "b"
            elif(currDirection == "r"):
                pathString+="rf"
                currDirection = "b"
            elif(currDirection == "b"):
                pathString+="f"
                currDirection = "b"
            elif(currDirection == "l"):
                pathString+="lf"
                currDirection = "b"
    prev = node
# Display output
# plt.imshow(output_rgb)
# plt.axis("off")
print(pathString)
plt.imshow(scaled_crop_image)
plt.show()

