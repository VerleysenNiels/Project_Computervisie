import cv2
import numpy as np

def imshow(img, name='Image', norm=False, resize=False):
    """Display an image

    Arguments:
        img  -- The image to be shown. If the dtype is `float` or `signed`,
            the `norm` should be set to true


    Keyword Arguments:
        name {str} -- The name of the window (default: {'Image'})
        norm {bool} -- Normalize the image between 0 and 255 (default: {False})
        resize {bool} -- Resize the image to be 500px wide (default: {False})
    """

    if resize:
        img = cv2.resize(
            img, (0, 0),
            fx=500 / img.shape[0],
            fy=500 / img.shape[0],
            interpolation=cv2.INTER_NEAREST)
    if norm:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow(name, img.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawLines(image, lines):
    #Lijnen tekenen
    for i in lines:
        print(i)
        for rho, theta in i:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0+1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 -1000*(a))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 2)

    return image