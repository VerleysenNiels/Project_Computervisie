import cv2


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
