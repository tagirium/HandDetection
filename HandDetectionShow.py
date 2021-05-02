import cv2
import mediapipe as mp
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
from statistics import median
'''devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
	IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volume_min = volume.GetVolumeRange()[0]
volume_max = volume.GetVolumeRange()[1]'''
#

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
ptime = time.time()
while True:
	success, img = cap.read()
	h, w, c = img.shape
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	res = hands.process(imgRGB)
	if res.multi_hand_landmarks:
		for handLMs in res.multi_hand_landmarks:
			lms = handLMs.landmark
			cv2.circle(img, (int(lms[4].x * w), int(lms[4].y * h)), 15, (255, 0, 255), cv2.FILLED)
			cv2.circle(img, (int(lms[8].x * w), int(lms[8].y * h)), 15, (255, 0, 255), cv2.FILLED)
			dist_x = abs(int(int(lms[4].x * w) - int(lms[8].x * w)))
			dist_y = abs(int(lms[4].y * h) - int(lms[8].y * h))
			pic_size_x = np.interp(dist_x, [15, 125], [1, 2])
			pic_size_y = np.interp(dist_y, [15, 125], [1, 2])
			cv2.line(img,
					 (int(lms[4].x * w), int(lms[4].y * h)), (int(lms[8].x * w), int(lms[8].y * h)), (255, 0, 255), 3)
			mpDraw.draw_landmarks(img, handLMs, mpHands.HAND_CONNECTIONS)
			img = cv2.resize(img, (int(w*pic_size_x), int(h*pic_size_y)))
	cTime = time.time()
	fps = str(int(1 / (cTime - ptime)))
	ptime = cTime
	cv2.putText(img, fps, (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 0), 3)
	cv2.imshow('Image', img)
	cv2.waitKey(1)
