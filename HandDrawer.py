import cv2
import HandDetectionModule as hdm
import time
import numpy as np
import os
folderPath = 'Header'
myList = os.listdir(folderPath)

drawColor = (255, 0, 255)
brushThickness = 15
xp, yp = 0, 0

overlays = []
for imPath in myList:
	img = cv2.imread(f'{folderPath}/{imPath}')
	overlays.append(img)
header = overlays[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

detector = hdm.HandDetector(det_confidence=0.85, max_num_hands=1)


while True:
	# 1. Gain image from camera
	success, img = cap.read()
	img = cv2.flip(img, 1)
	# 2. Find hand landmarks
	img = detector.findHands(img)
	lmList = detector.findPosition(img, draw=False)

	if len(lmList) != 0:

		x1, y1 = lmList[8][1:]
		x2, y2 = lmList[12][1:]
		# 3. Check which fingers are up
		fingers = detector.fingersUp()
		# 4. If 2 fingers are up - selection mode
		if fingers[1] and fingers[2]:
			xp, yp = 0, 0
			cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)
			if y1 < 199:
				if 230 < x1 < 330:
					header = overlays[1]
					drawColor = (0, 0, 0)
				elif 360 < x1 < 540:
					header = overlays[2]
					drawColor = (0, 0, 255)
				elif 590 < x1 < 780:
					header = overlays[3]
					drawColor = (0, 255, 0)
				elif 840 < x1 < 1050:
					header = overlays[4]
					drawColor = (255, 0, 0)
		# 5. If index finger is up - drawing mode
		if fingers[1] and not fingers[2]:

			cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
			if xp == yp == 0:
				xp, yp = x1, y1
			cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
			xp, yp = x1, y1

	imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
	_, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
	imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
	img = cv2.bitwise_and(img, imgInv)
	img = cv2.bitwise_or(img, imgCanvas)
	# only transparent colors
	# img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
	# Setting the header image
	img[0:199, 0:1280] = header
	# Optional circles above header, slows program a bit
	'''for lm in lmList:
		cv2.circle(img, (lm[1], lm[2]), 10, (255, 0, 255), cv2.FILLED)'''
	cv2.imshow('Image', img)
	cv2.waitKey(1)
