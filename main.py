import maze_to_string as solver
import cv2
import matplotlib.pyplot as plt
# Initialize the camera
camera = cv2.VideoCapture(0)

# Read a frame from the camera
return_value, image = camera.read()

# Save the captured image
cv2.imwrite('opencv.png', image)

# Release the camera
camera.release()
path, maze = solver.calculate("maze.jpeg")
print(path)
plt.imshow(maze)
plt.show()