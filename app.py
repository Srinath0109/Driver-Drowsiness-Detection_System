import cv2
from detector import DrowsinessDetector

def main():
    detector = DrowsinessDetector()

    print("🚗 Driver Drowsiness Detection System Started...")
    detector.start_detection()

if __name__ == "__main__":
    main()
