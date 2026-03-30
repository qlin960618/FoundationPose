#!/usr/bin/env python3
import cv2


def main():
  cap = cv2.VideoCapture(5)
  if not cap.isOpened():
    raise SystemExit("Failed to open webcam at index 0")

  print("Showing webcam feed from index 0. Press 'q' or Ctrl+C to exit.")

  try:
    while True:
      ok, frame = cap.read()
      if not ok:
        raise SystemExit("Failed to read frame from webcam")

      cv2.imshow("test_webcam", frame)
      if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
  except KeyboardInterrupt:
    pass
  finally:
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
