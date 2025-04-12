import maze_to_string as solver
import cv2
import serial
import matplotlib.pyplot as plt
import time
import requests

arduino_ip = "192.168.128.254"  # replace with actual IP
message = "Hello_Arduino"

path, maze = solver.calculate("maze.jpeg")
print(path)
path = "rrrr"
path+="\r"
# def send_string(data):
#     arduino.write(data.encode())
#     print(f"Sent: {data}")
#     time.sleep(1)

# def receive_string():
#     while True:
#         if arduino.in_waiting > 0:
#             received = arduino.readline().decode().strip()
#             print(f"Received from Arduino: {received}")
#             return received

# arduino = serial.Serial(port='COM6', baudrate=9600, timeout=1)
# send_string(path)
try:
    response = requests.get(f"http://{arduino_ip}/?msg={path}")
    print("Response from Arduino:", response.text)
except Exception as e:
    print("Error:", e)
time.sleep(2)   

# received_data = receive_string