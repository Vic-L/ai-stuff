import cv2

car_detector = cv2.CascadeClassifier('car_detector.xml')
pedestrian_detector = cv2.CascadeClassifier('pedestrian.xml')

## IMAGE
# img = cv2.imread('car.jpg')
# img = cv2.imread('traffic.jpg')

# grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cars = car_detector.detectMultiScale(grayscale_img)


## VIDEO
# video = cv2.VideoCapture('teslacam.mov')
video = cv2.VideoCapture('motorcycle.mov')

for i in range(1000):
  status, frame = video.read()
  grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  # DETECT
  cars = car_detector.detectMultiScale(grayscale_img, scaleFactor=1.5)
  pedestrians = pedestrian_detector.detectMultiScale(grayscale_img, scaleFactor=1.5)

  # DRAW RECT
  for (x, y, w, h) in cars:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
  for (x, y, w, h) in pedestrians:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

  cv2.imshow('test', frame)
  cv2.waitKey(1)

print("test")