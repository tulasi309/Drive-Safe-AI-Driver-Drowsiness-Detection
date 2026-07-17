from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame
import numpy as np
import time
from threading import Thread
from twilio.rest import Client  


pygame.mixer.init()


TWILIO_ACCOUNT_SID = "AC2e74c3c8ba4fe72676e7d4a8579ea4f5"
TWILIO_AUTH_TOKEN = "fa6f166da6021d6b13145ec9a8f23ed9"
TWILIO_PHONE_NUMBER = "+13082218843"
RECIPIENT_PHONE_NUMBER = "+916304113138"

def send_sms_alert():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body="🚨ALERT!your Driver has been drowsy!! Immediate action needed!",
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    print(f"📩 SMS sent: {message.sid}")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def play_alert_sound():
    pygame.mixer.music.load(r"C:\Users\prasa\OneDrive\Desktop\dectection\siren-alert-96052 (mp3cut.net).mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


def play_full_song():
    pygame.mixer.music.stop()
    pygame.mixer.music.load(r"C:\Users\prasa\OneDrive\Desktop\dectection\Fear.mp3")
    pygame.mixer.music.play(-1)  
    print("🎵 Full song is now playing...")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks (1).dat")  


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 



EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30  
alert_count = 0
consecutive_frames = 0
alert_triggered = False
sms_sent = False  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not access the webcam!")
        break

    frame = imutils.resize(frame, width=900)  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
        rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

       
        for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        
        left_x, left_y, left_w, left_h = cv2.boundingRect(leftEye)
        right_x, right_y, right_w, right_h = cv2.boundingRect(rightEye)

        eye_color = (0, 0, 255) if ear < EYE_AR_THRESH else (0, 255, 0)
        cv2.rectangle(frame, (left_x, left_y), (left_x + left_w, left_y + left_h), eye_color, 2)
        cv2.rectangle(frame, (right_x, right_y), (right_x + right_w, right_y + right_h), eye_color, 2)

        
        if ear < EYE_AR_THRESH:
            consecutive_frames += 1
            if consecutive_frames >= EYE_AR_CONSEC_FRAMES and not alert_triggered:
                cv2.putText(frame, "****** ALERT! ******", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "****** ALERT! ******", (50, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("🚨 ALERT! Driver is drowsy! 🚨")
                
                
                Thread(target=play_alert_sound).start()
                
                alert_triggered = True
                alert_count += 1


                
                if alert_count == 5:
                    print("Playing full song to keep driver awake...")
                    Thread(target=play_full_song).start()

                
                if alert_count == 7 and not sms_sent:
                    print("🚨 Sending SMS Alert... 🚨")
                    Thread(target=send_sms_alert).start()
                    sms_sent = True  

        else:
            consecutive_frames = 0
            alert_triggered = False

   
    cv2.namedWindow("Driver Drowsiness Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Driver Drowsiness Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Driver Drowsiness Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        pygame.mixer.music.stop()
        break


cap.release()
cv2.destroyAllWindows()
