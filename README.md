# Hand Tracking with OpenCV & MediaPipe

<div style="text-align: center;">
    <img src="images/banner.jpg" style="width:950px;height:450px;">
</div>

## Project Overview

This project is a real-time hand tracking application that uses OpenCV and MediaPipe. It consists of a main script for capturing video from the webcam and a module that handles the hand tracking functionality.

## Repository Structure

```
- Hand_Tracking.py
- hand_tracking_module.py
```

## Files Description

1. **Hand_Tracking.py**: Captures video from the webcam, processes each frame to detect hands, and displays the resulting video with hand landmarks and FPS.
2. **hand_tracking_module.py**: Contains a `handTracking` class that encapsulates the hand detection logic using MediaPipe. Provides methods to detect hands and find positions of the landmarks.

## Setup Instructions

Follow these steps to run the project on your local machine:

1. **Clone the Repository**

    ```sh
    git clone https://github.com/HoomKh/Hand_Tracking.git
    cd hand_tracking
    ```

2. **Install Dependencies**

    Ensure you have Python 3.x installed. Then, install the required packages:

    ```sh
    pip install opencv-python mediapipe
    ```

3. **Run the Application**

    Execute the main script:

    ```sh
    python Hand_Tracking.py
    ```

## Usage

- The application captures video from your webcam.
- It processes each frame to detect hands and draw landmarks on the detected hands.
- It displays the FPS on the screen.

## Example Code Snippets

### Hand_Tracking.py

```python
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
```

### hand_tracking_module.py

```python
import cv2
import mediapipe as mp

class handTracking:
    def __init__(self, mode=False, maxHands=2, modelComp=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        
        return lmList
```
