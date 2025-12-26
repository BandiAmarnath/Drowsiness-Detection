# Importing necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pygame  # Import pygame for sound playback
from keras.models import load_model


# Initialize pygame mixer for sound playback
pygame.mixer.init()
# Load the alarm sound
alarm_sound = pygame.mixer.Sound('alarm.wav')
model=load_model('models/2025_02_15_10_36_38.h5')

# Calculating eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Calculating mouth aspect ratio
def mouth_aspect_ratio(mou):
    X = dist.euclidean(mou[0], mou[6])
    Y1 = dist.euclidean(mou[2], mou[10])
    Y2 = dist.euclidean(mou[4], mou[8])
    Y = (Y1 + Y2) / 2.0
    mar = Y / X
    return mar

camera = cv2.VideoCapture(0)
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# Define constants for aspect ratios
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48
MOU_AR_THRESH = 0.75

COUNTER = 0
yawnStatus = False
yawns = 0
drowsiness_alert_played = False  # Flag to track if alert sound has played

# Initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Loop over capturing video
while True:
    ret, frame = camera.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_yawn_status = yawnStatus
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouEAR = mouth_aspect_ratio(mouth)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            cv2.putText(frame, "Eyes Closed ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Play alert sound if not already playing
                if not drowsiness_alert_played:
                    alarm_sound.play()  # Play the alarm sound
                    drowsiness_alert_played = True  # Set flag to prevent replaying

        else:
            COUNTER = 0
            drowsiness_alert_played = False  # Reset flag when eyes are open
            cv2.putText(frame, "Eyes Open ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mouEAR > MOU_AR_THRESH:
            cv2.putText(frame, "Yawning ", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yawnStatus = True
            output_text = "Yawn Count: " + str(yawns + 1)
            cv2.putText(frame, output_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            yawnStatus = False

        if prev_yawn_status == True and yawnStatus == False:
            yawns += 1

        cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "FATIGUE DETECTION SYSTEM", (370, 470), cv2.FONT_HERSHEY_COMPLEX, 0.6, (153, 51, 102), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
camera.release()
