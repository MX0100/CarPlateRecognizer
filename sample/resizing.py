import cv2

def image_acquisition(image_path):
    image = cv2.imread(image_path)
    return image

image = image_acquisition('C:/Users/WenKai/Desktop/License-Plate-Recognition-master/test/sampleImg.jpg')
cv2.waitKey(0)
cv2.destroyAllWindows()
def image_preprocessing(image):
    preprocessed_image = cv2.GaussianBlur(image, (5, 5), 0)
    return preprocessed_image

preprocessed_image = image_preprocessing(image)
cv2.imshow('Preprocessed Image', preprocessed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def image_resizing(image, width):
    height = int((image.shape[0] / image.shape[1]) * width)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image

resized_image = image_resizing(preprocessed_image, 600)
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()