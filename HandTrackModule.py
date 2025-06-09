import cv2
import mediapipe as mp
import time


#  1. Hand Detector Class Initialization
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=mode,
                                        max_num_hands=maxHands,
                                        min_detection_confidence=detectionCon,
                                        min_tracking_confidence=trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    # 2. Function to Find Hands and Draw Landmarks
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    #  3. Function to Get Positions of All Landmarks
    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList


# 🟢 4. Main Function to Run Hand Tracking
def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        # Print position of thumb tip (landmark ID 4)
        if lmList:
            print(lmList[4])
        # FPS Calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS on the screen
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
