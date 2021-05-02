import cv2
import mediapipe as mp
import time
import numpy as np


class HandDetector:
	def __init__(self, max_num_hands=2, mode=False, det_confidence = 0.5, track_conf = 0.5):
		self.mode = mode
		self.max_num_hands = max_num_hands
		self.det_confidence = det_confidence
		self.track_conf = track_conf

		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(self.mode, self.max_num_hands,
										self.det_confidence, self.track_conf)

		self.mpDraw = mp.solutions.drawing_utils

		self.tipIds = [4, 8, 12, 16, 20]

	def findHands(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.res = self.hands.process(imgRGB)
		if self.res.multi_hand_landmarks:
			for handLMs in self.res.multi_hand_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img, handLMs, self.mpHands.HAND_CONNECTIONS)

		return img

	def findPosition(self, img, hand_num=0, draw=True):
		self.lmList = []
		h, w, c = img.shape
		if self.res.multi_hand_landmarks:
			myHand = self.res.multi_hand_landmarks[hand_num]
			lms = myHand.landmark
			for id, lm in enumerate(lms):
				cx, cy = int(lm.x * w), int(lm.y * h)
				self.lmList.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
		return self.lmList

	def fingersUp(self):
		fingers = []
		# Thumb
		# problem doen't check wether it's left or right hand
		if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
			fingers.append(True)
		else:
			fingers.append(False)
		# 4 Fingers
		for id in range(1, 5):
			if self.lmList[self.tipIds[0]][2] > self.lmList[self.tipIds[id] - 1][2]:
				fingers.append(True)
			else:
				fingers.append(False)
		return fingers

def main():
	pTime = 0
	cap = cv2.VideoCapture(0)
	detector = HandDetector(max_num_hands=1)
	while True:
		success, img = cap.read()
		cTime = time.time()
		img = detector.findHands(img)
		lmlist = detector.findPosition(img)
		if len(lmlist) != 0:
			print(lmlist[4])
		time.sleep(0.0001)
		fps = str(int(1 / (cTime - pTime)))
		pTime = cTime

		cv2.putText(img, fps, (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 0), 3)
		cv2.imshow('Image', img)
		cv2.waitKey(1)


if __name__ == '__main__':
	main()
