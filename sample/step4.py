import cv2

# Step 1: Image Acquisition
def image_acquisition(image_path):
    image = cv2.imread(image_path)
    cv2.imshow('Acquired Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

image = image_acquisition('C:/Users/WenKai/Desktop/License-Plate-Recognition-master/test/sampleImg.jpg')

# Step 2: Image Preprocessing (e.g., noise reduction)
def image_preprocessing(image):
    preprocessed_image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow('Preprocessed Image', preprocessed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return preprocessed_image

preprocessed_image = image_preprocessing(image)

# Step 3: Image Resizing
def image_resizing(image, width):
    height = int((image.shape[0] / image.shape[1]) * width)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow('Resized Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return resized_image

resized_image = image_resizing(preprocessed_image, 600)

# Step 4: Grayscale Conversion
def grayscale_conversion(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return gray_image

gray_image = grayscale_conversion(resized_image)
# Further steps can follow a similar pattern...
def edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    cv2.imshow('Edge Detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edges

edges = edge_detection(gray_image)