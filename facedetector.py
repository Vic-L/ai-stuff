import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# IMAGES
# for filename in ['luffy.png', 'ChristianBale.jpg']:
#   # READ IMG AND SHOW
#   img = cv2.imread(filename)

#   # DETECT FACE
#   grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

#   # DRAW RECT
#   for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

#   # SHOW
#   cv2.imshow('test', img)
#   cv2.waitKey()

# VIDEO & CAMERAS
webcam = cv2.VideoCapture(0)

for i in range(100):
  status, frame = webcam.read()
  grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

  for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

  cv2.imshow('test', frame)
  cv2.waitKey(100)

print("finish")
