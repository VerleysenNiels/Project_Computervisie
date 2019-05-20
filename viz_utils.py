import cv2
import numpy as np
import math_utils

roomsAndCoords = {
    'zaal_a': (421, 508),
    'zaal_b': (326, 508),
    "zaal_c": (240, 508),
    "zaal_d": (253, 426),
    "zaal_e": (394, 426),
    "zaal_f": (423, 342),
    "zaal_g": (324, 344),
    "zaal_h": (239, 344),
    "zaal_i": (263, 279),
    "zaal_j": (378, 215),
    "zaal_k": (419, 158),
    "zaal_l": (477, 126),
    "zaal_m": (238, 237),
    "zaal_n": (131, 275),
    "zaal_o": (32, 275),
    "zaal_p": (229, 126),
    "zaal_q": (314, 121),
    "zaal_r": (240, 41),
    "zaal_s": (395, 46),
    "zaal_1": (643, 509),
    "zaal_2": (741, 509),
    "zaal_3": (826, 510),
    "zaal_4": (810, 427),
    "zaal_5": (669, 425),
    "zaal_6": (644, 347),
    "zaal_7": (740, 345),
    "zaal_8": (828, 345),
    "zaal_9": (699, 280),
    "zaal_10": (684, 216),
    "zaal_11": (639, 164),
    "zaal_12": (583, 128),
    "zaal_13": (827, 237),
    "zaal_14": (933, 276),
    "zaal_15": (1034, 278),
    "zaal_16": (835, 127),
    "zaal_17": (753, 118),
    "zaal_18": (827, 40),
    "zaal_19": (674, 46)
}


def imshow(img, name='Image', norm=False, resize=True):
    """Display an image
    """
    if resize:
        img = cv2.resize(
            img, (0, 0),
            fx=640 / img.shape[0],
            fy=640 / img.shape[0],
            interpolation=cv2.INTER_NEAREST)
    if norm:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow(name, img.astype('uint8'))
    # cv2.destroyAllWindows()


def overlay_polygon(img, points, color=(255, 0, 0)):
    out = np.copy(img)
    pts = points.reshape((-1, 1, 2))
    out = cv2.polylines(out, [pts], True, color, 2)
    return out


def overlay_lines(img, lines):
    """Overlay lines in polar coordinates on an image
    """
    out = np.copy(img)
    for i in lines:
        for rho, theta in i:
            x1, y1, x2, y2 = math_utils.to_carthesian(rho, theta)
            cv2.line(out, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return out


def overlay_lines_cartesian(img, lines):
    """Overlay lines in Cartesian coordinates on an image
    lines: array containing (x1, y1, x2, y2) quadruplets
    """
    out = np.copy(img)
    for line in lines:
        cv2.line(out, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
    return out


def overlay_points(img, points):
    """Overlay points on an image
    """
    out = np.copy(img)
    for i, point in enumerate(points):
        out = cv2.circle(out,
                         tuple(point),       # center
                         img.shape[0]//200,  # radius
                         ((47 * i) % 255, (67 * i) %
                          255, (97 * i) % 255),  # random color
                         img.shape[0]//200,
                         lineType=cv2.FILLED
                         )
    return out


def draw_path_line(img, room, nextroom):
    cv2.line(img, roomsAndCoords[room],
             roomsAndCoords[nextroom], (255, 0, 0), 2)


def process_gopro_video(frame, board_w, board_h):
    dims = (board_w, board_h)
    objp = np.zeros((dims[0]*dims[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:dims[0], 0:dims[1]].T.reshape(-1, 2)

    #Calib_M
    C = np.array([[7.2337882890945207e+02, 0., 6.4226033453805235e+02], [0.,
          7.2844995950341502e+02, 3.2297129949442024e+02], [0., 0., 1.]])
    D = np.array([-2.7971075073202351e-01, 1.2737835217024596e-01,
          5.5264049900636148e-04, -2.4709811526299534e-04,
          -3.7787805887358195e-02])

    #Calib_W
    #C = np.array([[5.6729034524746328e+02, 0., 6.3764777940570559e+02], [0.,
    #                                                                     5.7207768469558505e+02,
    #                                                                     3.3299427011674493e+02], [0., 0., 1.]])
    #D = np.array([-2.4637408439446815e-01, 7.6662428015464898e-02,
    #              -2.7014001885212116e-05, -3.1925229062179259e-04,
    #

    im_rect = cv2.undistort(frame, C, D, None)
    return im_rect