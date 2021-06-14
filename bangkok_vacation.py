import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

cap = cv2.VideoCapture(0)
time.sleep(2)

bg = 0

for i in range(60):
  ret, bg = cap.read()

bg = np.flip(bg, axis=1)

cv2.imshow('bg', bg)

while (cap.isOpened()):
  ret, img = cap.read()

  if not ret:
    break

  img = np.flip(img, axis=1)

  cv2.imshow('img', img)

  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower_red = np.array([0, 0, 0])
  upper_red = np.array([105, 105, 105])
  mask1 = cv2.inRange(hsv, lower_red, upper_red)
  
  lower_red_2 = np.array([128, 128, 128])
  upper_red_2 = np.array([169, 169, 169])
  mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

  mask = mask1 + mask2

  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
  mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

  mask2 = cv2.bitwise_not(mask)

  res1 = cv2.bitwise_and(img, img, mask=mask2)
  res2 = cv2.bitwise_and(bg, bg, mask=mask)

  final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

  output_file.write(final_output)

  cv2.imshow('Magic', final_output)
  cv2.waitKey(1)

cap.release()
final_output.release()
cv2.destroyAllWindows()