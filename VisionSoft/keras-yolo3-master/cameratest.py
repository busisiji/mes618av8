
import cv2


cap = cv2.VideoCapture(0)
while (cv2.waitKey(100) & 0xff != ord('q')):
    ret, frame = cap.read()
    # img = Image.frombytes("RGB", [frame.shape[1], frame.shape[0]], frame.tobytes())
    # img.show("test")
    cv2.imshow('img', frame)
cap.release()
cv2.destroyAllWindows()
