import label_util
import feature_detection
import io_utils
import csv
import numpy as np
import cv2

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

with open("csv_corners\corners_15.csv", mode = "r") as cornerlabels:
    corner_reader = csv.reader(cornerlabels, delimiter=";")
    with open("csv_detection_perf\perf_15.csv", mode="w") as perf:
        perf_writer = csv.writer(perf, delimiter=";")
        for row in corner_reader:
            alldigits = True
            for coord in row[1:]:
                if not coord.isdigit:
                    alldigits = False
            if alldigits and len(row) >= 9:
                path1 = "images/zalen/zaal_15/" + row[0]
                print(path1)
                img = io_utils.imread(path1)
                pts_lbl = np.array([(int(row[1]), int(row[2])), (int(row[3]), int(row[4])),
                                    (int(row[5]), int(row[6])), (int(row[7]), int(row[8])), ])
                #label_util.draw_quad(pts_lbl, img, blue)
                pts_det, _ = feature_detection.detect_perspective(img)
                if len(pts_det) >= 4:
                    #label_util.draw_quad(pts_det, img, green)
                    inters = label_util.calculate_intersection(pts_lbl, pts_det)
                    union = label_util.calculate_union(pts_lbl, pts_det)
                    if union != 0:
                        rat = inters/union
                        #print("\tIntersection: ", end=" ")
                        #print(inters)
                        #print("\tUnion: ", end=" ")
                        #print(union)
                        #print("\tRatio (intersection/union): ", end=" ")
                        #print(rat*100, end=" ")
                        #print("%")
                        perf_writer.writerow([row[0], rat])
                    else:
                        print("Zero union.")
                else:
                    print("Not enough points detected.")
                #cv2.imshow(row[0], img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
            elif not alldigits:
                print("Could not read corner coordinates.")
            elif len(row) < 2:
                print("")
            elif len(row) < 9:
                print("Not enough corner coordinates from labels.")
            else:
                print("I don't know what's wrong.")

