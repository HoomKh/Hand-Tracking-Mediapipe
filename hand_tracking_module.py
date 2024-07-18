import cv2
import mediapipe as mp
import time

class handTracking():

    def __init__(self, mode = False, maxHands = 2, modelComp = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp, self.detectionCon, self.trackCon )
        self.mpDraw = mp.solutions.drawing_utils



    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results .multi_hand_landmarks:
            for hand in self.results .multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)

        return img


    def findPosition(self, img, handNo = 0, draw = True):

        lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        return lm_list


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handTracking()

    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lm_list = detector.findPosition(img)
        if len(lm_list) != 0:
            print(lm_list[4])

        curTime = time.time()
        fps = 1 / (curTime - pTime)
        pTime = curTime
        cv2.putText(img, str(int(fps)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()