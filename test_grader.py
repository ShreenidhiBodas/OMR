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

cv2.imshow('Preprocessed',edged)
#find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#contour of the document i.e. answer sheet
docCnt = None
paper_contours = image.copy()
if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True) #approximating the contour to 2% of the found contour.

        if len(approx) == 4:
            docCnt = approx
            cv2.drawContours(paper_contours, [cnt], -1, (0,0,255), 3)
            break

paper = four_point_transform(image, docCnt.reshape((4, 2)))
warped = four_point_transform(gray, docCnt.reshape((4, 2)))

thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow('Paper Contours Extracted', paper_contours)
cv2.imshow('Paper Extracted Original' ,paper)
cv2.imshow('Paper Extracted Grayscale' ,warped)
cv2.imshow('Threshold' ,thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#after thresholding image, we again use contour extraction to find bubbles from the answer sheet
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

questionContours = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    aspect_ratio = w / float(h)

    if w >= 20 and h >= 20 and aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
	    questionContours.append(c)

for cnt in questionContours:
    cv2.drawContours(paper, [cnt], 0, (0, 0, 255), 3)

cv2.imshow('image', paper)
# cv2.waitKey(0)

question_cnts = contours.sort_contours(questionContours, method='top-to-bottom')[0]
correct = 0

for (q, i) in enumerate(np.arange(0, len(question_cnts), 5)):
    cnts = contours.sort_contours(question_cnts[i: i+5])[0]
    bubbled = None

    for (j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(mask, [c], -1, 255, -1)

        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        if bubbled is None or bubbled[0] < total:
            bubbled = (total, j)

    color = (0, 0, 255)
    k = ANSWER_KEY[q]

    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
    
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

print(correct)
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Exam", paper)
cv2.waitKey(0)