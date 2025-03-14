import cv2

def image_acquisition(image_path):
    image = cv2.imread(image_path)
    return image

image = image_acquisition('C:/Users/WenKai/Desktop/License-Plate-Recognition-master/test/sampleImg.jpg')
cv2.imshow('Acquired Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
def image_preprocessing(image):
    preprocessed_image = cv2.GaussianBlur(image, (5, 5), 0)
    return preprocessed_image

# 示例
preprocessed_image = image_preprocessing(image)
cv2.imshow('Preprocessed Image', preprocessed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
