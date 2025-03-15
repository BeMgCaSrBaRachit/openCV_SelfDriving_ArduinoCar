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
# plt.imshow(img_original)
# cv2.imshow("original", cv_image)
# img = img_original.mean(axis=2, keepdims = True)/255
# img = np.concatenate([img_original*3], axis = 2)
img = img_original
print('Program Running......')
vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]

n,m,d = img.shape
edges = np.zeros((n,m))

# def clamp(n, minn, maxn):
#     return max(minn, min(n, maxn))

for row in range(3,n-2):
    for col in range(3,m-2):
        local_pixels = img[row-1 : row+2, col-1: col+2, 0] #This 0 is here because my filteration matrix is 3D
        vertical_transformed_pixels = vertical_filter*local_pixels
        vertical_score = vertical_transformed_pixels.sum()/4

        horizontal_transformed_pixels = horizontal_filter*local_pixels
        horizontal_score = horizontal_transformed_pixels.sum()/4

        edge_score = math.sqrt(math.pow(horizontal_score,2) + math.pow(vertical_score,2))
        edges[row][col] = 3*edge_score

# cv2.imshow("edging??", edges)
# cv2.imshow("outline",edges)

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

print(minX, minY, maxX, maxY)
# cv2.rectangle(cv_image, (minX,minY), (maxX,maxY), (0,0,255), 3)
crop_image = img_original[minY:maxY, minX:maxX]
plt.imshow(crop_image)
# cv2.imshow("boundary",cv_image)
# n1 = maxX-minX-5
# m1 = maxY-minY-5
# redMin = [2000,2000]
# redMax = [0,0]
# greenMin = [sys.maxsize,sys.maxsize]
# greenMax = [0,0]
# hsv_frame = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)
# for rows in range(0,n1):
#     for col in range(0,m1):
#         val = hsv_frame[rows][col]
#         if(val[0] >= 118 and val[0] <= 120):
#             redMin[0] = min(col, redMin[0])
#             redMin[1] = min(rows, redMin[1])
#             redMax[0] = max(col, redMax[0])
#             redMax[1] = max(rows, redMax[1])
#         elif(val[0] >= 60 and val[0] <= 61):
#             greenMin[0] = min(col, greenMin[0])
#             greenMin[1] = min(rows, greenMin[1])
#             greenMax[0] = max(col, greenMax[0])
#             greenMax[1] = max(rows, greenMax[1])
# print(redMin, redMax)
# cv2.rectangle(cv_image, (redMin), (redMax), (255,0,0), 3)
# cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
# cv2.imshow("red_BOX", cv_image)
# plt.imshow(scaled_crop_image)
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

# Convert grayscale masks to binary
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes
output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

red_box, green_box = None, None

for cnt in contours_red:
    x, y, w, h = cv2.boundingRect(cnt)
    red_box = (x, y, w, h)  # Store red bounding box
    cv2.rectangle(output_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box

for cnt in contours_green:
    x, y, w, h = cv2.boundingRect(cnt)
    green_box = (x, y, w, h)  # Store green bounding box
    cv2.rectangle(output_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

# Determine relative position
if red_box and green_box:
    rx, ry, rw, rh = red_box
    gx, gy, gw, gh = green_box

    if gx + gw < rx:  
        position = "Left"
    elif gx > rx + rw:  
        position = "Right"
    elif gy + gh < ry:  
        position = "Above"
    elif gy > ry + rh:  
        position = "Below"
    elif gx + gw < rx and gy + gh < ry:
        position = "Top-left"
    elif gx > rx + rw and gy + gh < ry:
        position = "Top-right"
    elif gx + gw < rx and gy > ry + rh:
        position = "Bottom-left"
    elif gx > rx + rw and gy > ry + rh:
        position = "Bottom-right"
    else:
        position = "Overlapping"

    print(f"The green box is {position} relative to the red box.")

# Convert back to RGB for matplotlib
output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
scaled_crop_image = cv2.resize(crop_image, (5,5), cv2.INTER_LINEAR_EXACT)
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
                source = (rows, col)
            elif b == 1:
                destination = (rows, col)
Matrix = np.array(final_matrix)
# print(Matrix)
print(source, destination)

grid = Grid(matrix=Matrix)
start = grid.node(source[1], source[0])
end = grid.node(destination[1], destination[0])

finder = AStarFinder(diagonal_movement=0)

path,runs = finder.find_path(start, end, grid)
print(path)
# Display output
plt.imshow(output_rgb)
plt.axis("off")
plt.show()