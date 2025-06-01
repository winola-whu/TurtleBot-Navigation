import cv2

img = cv2.imread("/home/winolah/ros2_ws/src/cube.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def show_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("HSV at ({}, {}): {}".format(x, y, hsv[y, x]))

cv2.imshow("Cube Image", img)
cv2.setMouseCallback("Cube Image", show_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

