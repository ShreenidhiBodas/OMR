import cv2
from imutils import contours
import argparse
import numpy as np
from four_point_transform import four_point_transform
import imutils

# create argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help="path to the image")
args = vars(ap.parse_args())

#define the answer key
#answer key is a dictionary with zero based indexing fot the question and answers
ANSWER_KEY = { 0:1, 1:4, 2:0, 3:3, 4:1 }

#preprocess the image
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 75, 200)

#find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#contour of the document i.e. answer sheet
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, cv2.contourArea, reverse = True)

    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True) #approximating the contour to 2% of the found contour.

        if len(approx) == 4:
            docCnt = approx
            break

paper = four_point_transform(image, docCnt.reshape((4, 2)))
warped = four_point_transform(image, docCnt.reshape((4, 2)))

thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
