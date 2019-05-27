import cv2
import csv
import numpy as np

imname = "Groundplan"

#First the numbered rooms, then the lettered rooms, then room II


def draw_point (event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 3, (0, 255, 0), thickness=-1)
        cv2.imshow(imname, img)
        if len(points) < 1:
            tel = "1"
        elif points[len(points)-1][0].isdigit():
            t = int(points[len(points)-1][0])
            if t == 19:
                tel = "A"
            else:
                tel = str(t+1)
        else:
            t = points[len(points)-1][0]
            if t == "V":
                tel = "II"
            elif t == "S":
                tel = "V"
            else:
                if len(t) == 1:
                    tel = chr(ord(t) + 1)
                else:
                    tel = "TMR" #Too Many Rooms
        print(tel)
        row = np.array([tel, str(x), str(y)])
        points.append(row)


with open("rooms.csv", mode = "w") as pandc:
    corner_writer = csv.writer(pandc, delimiter=";", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
    img = cv2.imread("images/groundplan.png")
    cv2.namedWindow(imname)
    pts = []
    cv2.setMouseCallback(imname, draw_point, pts)
    cv2.imshow(imname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(pts)
    for room in pts:
        if len(room) >= 3:
            corner_writer.writerow([room[0], room[1], room[2]])
