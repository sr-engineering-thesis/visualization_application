import cv2

def upscale_image(image, scale_factor=2, interpolation=cv2.INTER_CUBIC):

    h, w = image.shape[:2]
    upscaled_image = cv2.resize(image, (w * scale_factor, h * scale_factor), interpolation=interpolation)

    return upscaled_image

