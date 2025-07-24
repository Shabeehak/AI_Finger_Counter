import cv2 as cv
import mediapipe as mp
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
drawing = mp.solutions.drawing_utils

# GET HANDLAND MARK
def getHandlandMarks(img, draw):
    lmlist = []
    frameRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    handsDetected = hands.process(frameRGB)
    label = None
    if handsDetected.multi_hand_landmarks:
        for landmarks in handsDetected.multi_hand_landmarks:
            for id, lm in enumerate(landmarks.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append((id, cx, cy))
                # print(lmlist)
        if draw:
            drawing.draw_landmarks(img, landmarks, mpHands.HAND_CONNECTIONS)
    if handsDetected.multi_handedness:
        label = handsDetected.multi_handedness[0].classification[0].label
    return lmlist, label

# FINGER COUNTING
def fingerCount(lmlist, label):
    count = 0
    if len(lmlist) != 0:
        # Thumb: check direction based on hand
        if label == 'Right':
            if lmlist[4][1] < lmlist[3][1]:
                count += 1
        elif label == 'Left':
            if lmlist[4][1] > lmlist[3][1]:
                count += 1

        # Fingers: up if tip is above PIP joint
        if lmlist[8][2] < lmlist[6][2]:   # Index
            count += 1
        if lmlist[12][2] < lmlist[10][2]: # Middle
            count += 1
        if lmlist[16][2] < lmlist[14][2]: # Ring
            count += 1
        if lmlist[20][2] < lmlist[18][2]: # Pinky
            count += 1

    return count


# CAMERA SETUP
cam = cv.VideoCapture(0)
while True:
    success, frame = cam.read()
    if not success:
        print('Camera not connected...!')
        continue
    frame = cv.flip(frame,1)
    lmlist, label = getHandlandMarks(frame, draw=True)
    if lmlist:
        # print(lmlist)
        fc = fingerCount(lmlist=lmlist, label=label)
        # print(fc)
        cv.rectangle(frame, (400,10), (600,250), (0,0,0),-1)
        cv.putText(frame, str(fc), (400,240), cv.FONT_HERSHEY_PLAIN, 20, (0,255,255), 30)
    cv.imshow('AI Finger Counter', frame)
    if cv.waitKey(1)==ord('q'):
        break
cam.release()
cv.destroyAllWindows()
hands.close()