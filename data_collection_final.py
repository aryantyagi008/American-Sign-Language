import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

c_dir = 'A'
base_path = "C:\\Users\\aryan\\Desktop\\ASL\\Final Project\\Source Code\\AtoZ_3.1"
count = len(os.listdir(os.path.join(base_path, c_dir)))

offset = 15
step = 1
flag = False
suv = 0

white = np.ones((400, 400), np.uint8) * 255
cv2.imwrite("./white.jpg", white)

try:
    while True:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands, frame = hd.findHands(frame, draw=False, flipType=True)  # Corrected line: `hands` is now a list
        white = cv2.imread("./white.jpg")

        if hands:
            hand = hands[0]  # Get the first hand (if more than one hand is detected)
            x, y, w, h = hand['bbox']  # Access the bounding box of the hand

            # Corrected line here
            image = np.array(frame[max(0, y - offset):y + h + offset, max(0, x - offset):(x + w + offset)])

            handz, imz = hd2.findHands(image, draw=True, flipType=True)
            if handz:
                hand = handz[0]
                pts = hand['lmList']
                os_offset = ((400 - w) // 2) - 15
                os1_offset = ((400 - h) // 2) - 15

                # Draw lines for skeleton
                for t in range(0, 4):
                    cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                for t in range(5, 8):
                    cv2.line(white, (pts[t][0] + os_offset, pts[t][1] + os1_offset), (pts[t + 1][0] + os_offset, pts[t + 1][1] + os1_offset), (0, 255, 0), 3)
                # Repeat for the rest...

                skeleton1 = np.array(white)

                # Draw circles for joints
                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os_offset, pts[i][1] + os1_offset), 2, (0, 0, 255), 1)

                cv2.imshow("Skeleton", skeleton1)

        frame = cv2.putText(frame, "dir=" + str(c_dir) + "  count=" + str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)

        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:  # esc key
            break

        if interrupt & 0xFF == ord('n'):
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) == ord('Z') + 1:
                c_dir = 'A'
            flag = False
            count = len(os.listdir(os.path.join(base_path, c_dir)))

        if interrupt & 0xFF == ord('a'):
            flag = not flag
            suv = 0 if flag else suv

        if flag and suv < 180:
            if step % 3 == 0:
                output_dir = os.path.join("D:\\ASL\\Sign-Language-To-Text-and-Speech-Conversion\\AtoZ_3.2", c_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                cv2.imwrite(os.path.join(output_dir, f"{count}.jpg"), skeleton1)
                count += 1
                suv += 1
            step += 1

except Exception as e:
    print("Error:", traceback.format_exc())

finally:
    capture.release()
    cv2.destroyAllWindows()
