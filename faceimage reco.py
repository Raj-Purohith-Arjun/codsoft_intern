import cv2
import dlib
# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image or video stream
image = cv2.imread('face_image.jpg')  # Replace with the path to your image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Load the pre-trained face recognition model
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load an image with known faces
known_image = dlib.load_rgb_image('known_face.jpg')  # Replace with the path to your image
known_face_descriptor = face_recognizer.compute_face_descriptor(known_image)

# Load another image with an unknown face
unknown_image = dlib.load_rgb_image('unknown_face.jpg')  # Replace with the path to your image
unknown_face_descriptor = face_recognizer.compute_face_descriptor(unknown_image)

# Calculate the Euclidean distance between known and unknown face descriptors
distance = dlib.distance(known_face_descriptor, unknown_face_descriptor)

if distance < 0.6:
    print("Recognized: Known face")
else:
    print("Unknown face")
