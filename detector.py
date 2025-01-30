import cv2
import dlib
from scipy.spatial import distance as dist
from utils import eye_aspect_ratio, get_eye_coordinates
from sound_alert import play_alert_sound

# Load Dlib's face detector and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor.dat")

class DrowsinessDetector:
    def __init__(self, eye_thresh=0.25, frame_thresh=20):
        self.eye_thresh = eye_thresh  # Eye aspect ratio threshold
        self.frame_thresh = frame_thresh  # Number of frames eyes must be closed
        self.counter = 0

    def start_detection(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            for face in faces:
                landmarks = landmark_predictor(gray, face)

                left_eye, right_eye = get_eye_coordinates(landmarks)
                left_EAR = eye_aspect_ratio(left_eye)
                right_EAR = eye_aspect_ratio(right_eye)
                avg_EAR = (left_EAR + right_EAR) / 2.0

                if avg_EAR < self.eye_thresh:
                    self.counter += 1
                    if self.counter >= self.frame_thresh:
                        play_alert_sound()
                        cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    self.counter = 0

                # Draw rectangles around eyes
                for (x, y) in left_eye + right_eye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
