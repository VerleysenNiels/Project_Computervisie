import math
import perspective

def angle_betw_lines (line1, line2):
    # equation as "y = ax + b"
    # a = (y2 - y1)/(x2 - x1)
    dx1 = (line1[1, 0] - line1[0, 0])
    dx2 = (line2[1, 0] - line2[0, 0])
    dy1 = (line1[1, 1] - line1[0, 1])
    dy2 = (line2[1, 1] - line2[0, 1])
    theta1 = math.pi/2
    theta2 = math.pi/2
    if dx1 != 0:
        theta1 = math.atan(dy1 / dx1)
    if dx2 != 0:
        theta2 = math.atan(dy2 / dx2)
    phi = theta1 - theta2
    while phi<0:
        phi += math.pi
    return phi


def pnt_above_line(pt, ln):
    # a point (xp, yp) lies above a line if (y1-y2)yp + (x1-x2)xp + (x1y2 - x2y1)> 0
    if ((ln[0][1] - ln[1][1]) * pt[1] + (ln[0][0] - ln[1][0]) * pt[0] +
       ((ln[0][0] * ln[1][1]) - (ln[1][0] * ln[0][1]))) > 0:
        print("above")
        return True
    print("below")
    return False


def pnt_below_line(pt, ln):
    # a point (xp, yp) lies above a line if (y1-y2)yp + (x1-x2)xp + (x1y2 - x2y1)> 0
    if ((ln[0][1] - ln[1][1]) * pt[1] + (ln[0][0] - ln[1][0]) * pt[0] +
       ((ln[0][0] * ln[1][1]) - (ln[1][0] * ln[0][1]))) < 0:
        print("below")
        return True
    print("above")
    return False


def pnt_left_line(pt, ln):
    # a point (xp, yp) lies left of a line if ((x2-x1)*(yp-y1))-((y2-y1)*(xp-x1)) > 0
    if (((ln[1][0] - ln[0][0]) * (pt[1] - ln[0][1])) - ((ln[1][1] - ln[0][1]) * (pt[0] - ln[0][0]))) > 0:
        print("left")
        return True
    print("right")
    return False


def pnt_right_line(pt, ln):
    # a point (xp, yp) lies left of a line if ((x2-x1)*(yp-y1))-((y2-y1)*(xp-x1)) > 0
    if (((ln[1][0] - ln[0][0]) * (pt[1] - ln[0][1])) - ((ln[1][1] - ln[0][1]) * (pt[0] - ln[0][0]))) < 0:
        print("right")
        return True
    print("left")
    return False


def point_inside_quad(pt, quad):
    if ((pnt_right_line(pt, (quad[0], quad[3]))) and
        (pnt_below_line(pt, (quad[0], quad[1]))) and
        (pnt_left_line(pt, (quad[1], quad[2]))) and
        (pnt_above_line(pt, (quad[3], quad[2])))):
        print("inside")
        return True
    print("outside")
    return False


def no_intersection(quad1, quad2):
    # still need to implement this
    quad1 = perspective.order_points(quad1)
    quad2 = perspective.order_points(quad2)
    # a point (xp, yp) lies above a line if (y1-y2)yp + (x1-x2)xp + (x1y2 - x2y1)> 0
    # quadrilateral1 above quadrilateral2
    if (((quad2[0, 1] - quad2[1, 1]) * quad1[3, 1] +
        (quad2[0, 0] - quad2[1, 0]) * quad1[3, 0] +
        ((quad2[0, 0]*quad2[1, 1])-(quad2[1, 0]*quad2[0, 1]))) > 0 and
        ((quad2[0, 1] - quad2[1, 1]) * quad1[2, 1] +
         (quad2[0, 0] - quad2[1, 0]) * quad1[2, 0] +
         ((quad2[0, 0] * quad2[1, 1]) - (quad2[1, 0] * quad2[0, 1]))) > 0):
        return True
    # quadrilateral2 above quadrilateral1
    elif (((quad1[0, 1] - quad1[1, 1]) * quad2[3, 1] +
           (quad1[0, 0] - quad1[1, 0]) * quad2[3, 0] +
           ((quad1[0, 0]*quad1[1, 1])-(quad1[1, 0]*quad1[0, 1]))) > 0 and
          ((quad1[0, 1] - quad1[1, 1]) * quad2[2, 1] +
           (quad1[0, 0] - quad1[1, 0]) * quad2[2, 0] +
           ((quad1[0, 0] * quad1[1, 1]) - (quad1[1, 0] * quad1[0, 1]))) > 0):
        return True
    # quadrilateral1 left of quadrilateral2

    # quadrilateral2 left of quadrilateral1
    return False