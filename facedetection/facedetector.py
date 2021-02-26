import cv2
import numpy

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_face_data = cv2.CascadeClassifier('smile.xml')

# IMAGES
# for filename in ['luffy.png', 'ChristianBale.jpg', 'escobar.jpg']:
#   # READ IMG AND SHOW
#   img = cv2.imread(filename)

#   # DETECT FACE
#   grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

#   # DRAW RECT
#   for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
#     face = img[y:y+h, x:x+w]
#     cv2.imshow('face', face)
#     cv2.waitKey(1000)
#     grayscale_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#     smile_coordinates = smile_face_data.detectMultiScale(grayscale_face, minNeighbors=80)

#     for (_x, _y, _w, _h) in smile_coordinates:
#       cv2.rectangle(img, (x + _x, y + _y), (x + _x + _w, y + _y + _h), (255, 0, 0), 4)

#   # SHOW
#   cv2.imshow('test', img)
#   cv2.waitKey(5000)

# VIDEO & CAMERAS
webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture('faces.mp4')

for i in range(100):
  status, frame = webcam.read()
  grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  face_coordinates = trained_face_data.detectMultiScale(grayscale_img, minNeighbors=10)

  for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    face = frame[y:y+h, x:x+w]
    cv2.imshow('face', face)
    cv2.waitKey(1000)
    grayscale_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    smile_coordinates = smile_face_data.detectMultiScale(grayscale_face, minNeighbors=20, scaleFactor=1.7)

    for (_x, _y, _w, _h) in smile_coordinates:
      cv2.rectangle(frame, (x + _x, y + _y), (x + _x + _w, y + _y + _h), (255, 0, 0), 4)

  cv2.imshow('test', frame)
  cv2.waitKey(10)

print("finish")
