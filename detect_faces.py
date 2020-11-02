# USAGE
# python detect_faces.py --face cascades/haarcascade_frontalface_default.xml --image images/test.png

# import the necessary packages
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="Path to where the face cascade model resides")
ap.add_argument("-i", "--image", required=True, help="Path to where the image file resides")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the face detector and detect faces in the image
faceDetector = cv2.CascadeClassifier(args["face"])


# Use the below code uncomment only when, if you are using OpenCV version 2.4

# faceRegions = faceDetector.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=5,
# 		minSize=(32, 32), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)


# Use the below code, if you are using OpenCv 3.0+
faceRegions = faceDetector.detectMultiScale(grayImage, scaleFactor=1.06, minNeighbors=5,
		minSize=(32, 32), flags=cv2.CASCADE_SCALE_IMAGE)

print("Total {} face(s) found on the image".format(len(faceRegions)))

# Now we are loop over the faces and draw a rectangle around each on the image face which we found.
for (x, y, w, h) in faceRegions:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the detected faces on screen
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)