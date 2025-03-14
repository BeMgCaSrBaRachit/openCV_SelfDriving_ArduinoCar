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
# cv2.imshow("original", cv_image)
img = img_original.mean(axis=2, keepdims = True)/255
img = np.concatenate([img_original*3], axis = 2)
print('Program Running......')

vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]
horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]

n,m,d = img.shape
edges = np.zeros((n,m))

def clamp(n, minn, maxn):
    return max(minn, min(n, maxn))

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

minX = sys.maxsize
minY = sys.maxsize

maxX  = -1
maxY = -1
maxVal = -1
for row in range(5,n-3): #constant X
    curr = edges[row][int(m/2)]
    next = edges[row+1][int(m/2)]
    if(next - curr >= 100):
        minY = min(minY,row)
    if(curr - next >= 100):
        maxY = max(maxY, row)

for col in range(5,m-3): #constant Y
    curr = edges[int(n/2)][col]
    next = edges[int(n/2)][col+1]
    if(next - curr >= 100):
        minX = min(minX,col)
    if(curr - next >= 100):
        maxX = max(maxX, col)

print(minX, minY, maxX, maxY)
cv2.rectangle(cv_image, (minX,minY), (maxX,maxY), (0,255,0), 3)
crop_image = img_original[minY:maxY, minX:maxX]
cv2.imshow("boundary",cv_image)
scaled_crop_image = cv2.resize(crop_image, (5,5), cv2.INTER_LINEAR)
# cv2.imshow("croppped",crop_image)
final_matrix = np.ones((5,5))
source = (0,0)
destination = (0,0)

for rows in range(0,5):
    for col in range(0,5):
        r = round(scaled_crop_image[rows][col][0]/255)
        g = round(scaled_crop_image[rows][col][1]/255)
        b = round(scaled_crop_image[rows][col][2]/255)
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
print(Matrix)
print(source, destination)

grid = Grid(matrix=Matrix)
start = grid.node(source[0], source[1])
end = grid.node(destination[0], destination[1])

finder = AStarFinder()

path,runs = finder.find_path(start, end, grid)
print(path)

plt.imshow(scaled_crop_image)
plt.show()
