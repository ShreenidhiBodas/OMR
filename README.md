# BubbleGrader

> BubbleGrader is a tool to analyse and detect the bubbles filled in OMR answer sheets and accordingly grade the answer sheet.  

## What is OMR?
OMR stands for **Optical Mark Recognition** is the process of automatically analyzing human-marked documents and interpreting their results.
  
## Tools and libraries used  
- opencv
- numpy
- imutils  
  
To install all these dependencies, use `python -m pip install requirements.txt`.  
**Requirements.txt** is provided with this repository.  

## Steps
1. Detect the exam paper from the image
2. Apply a perspective transform to extract the top-down, birds eye view of the exam sheet.
3. Extract the set of bubbles from the perspective transformed exam sheet.
4. Sort the bubbles from the exam sheet into rows. 1 row corresponding to 1 question.
4. Determine the marked bubbles from each row.
5. Cross-check the marked answer with the answer key.
6. Repeat this process for all rows.  
  
  
## Implementation  
The line 16 from the file *test_grader.py* defines the answer key for our exam.  
```python
ANSWER_KEY = { 0:1, 1:4, 2:0, 3:3, 4:1 }
```  
It is basically a dictionary.  
Right from line 18 to 25 we preprocess the image.
```python
if args["crop"] == 1:
    image = imutils.resize(cv2.imread(args["image"]), width=700)
else:
    image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 75, 200)
```  
By preprocessing, I mean, resizing, blurring, converting the image to grayscale and edge detection.  
Each of these processes help reduce the complexity of the further operations to be done on the image.  
>Images of the preprocessing done.  
  
The next step is to identify the exam sheet present in the image. Here, we use contours to find the exam sheet.  
```python
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
```  
After this, the cnts variable stores a list of contours. Now, in order to identify the exam paper, (which will be a rectangle) we loop through the  contours to find a contour having **4** corner points. This contour will correspond to the exam sheet. The code to achieve this is  
```python
for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True) #approximating the contour to 2% of the found contour.

        if len(approx) == 4:
            docCnt = approx
            break
```  
The function `cv2.approxPolyDP` approximates the contour ``cnt`` to 2% of the original contour. For more info about this function, visit [here](https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html).  
Variable `docCnt` now has the contour of the exam paper.
> Image of exam contour detected.  
  
Now, we apply perspective transform on the image to get a birds eye view.
```python
warped = four_point_transform(gray, docCnt.reshape((4, 2)))
```  
After this, we threshold the image using Binary / Otsu threshold
> Thresholded image  
  

Binarization again allows us to find contours from the image. This time, the contours will be the bubbles from the image. We append all such contours in a list called `questionContours`  
```python
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    aspect_ratio = w / float(h)

    if w >= 20 and h >= 20 and aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
	    questionContours.append(c)
```  
Here, we calculate the bounding box around each contour and check whether it is actually a bubble by checking aspect ratio and width-height. After validations, we append the contour to `questionContours`.  
```python
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

```  
This arranges the contours in rows, each row corresponding to each question and identifies the filled bubble. `total = cv2.countNonZero(mask)` counts number of non zero pixels in the contour. After this, we compare the filled bubble index to that of the correct answer in our `ANSWER_KEY` and print the corresponding result and draw the appropriate contour on the image.  
  
> Final image.