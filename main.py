import cv2
import numpy as np
import pytesseract
import os


def process_image(image_path):
    # Step 1: Image Acquisition
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not read image from the given path.")

    # Display and save the original image
    cv2.imshow('Original Image', image)
    cv2.imwrite('original_image.jpg', image)
    cv2.waitKey(0)

    # Step 2: Image Preprocessing (noise reduction)
    preprocessed_image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow('Preprocessed Image', preprocessed_image)
    cv2.imwrite('preprocessed_image.jpg', preprocessed_image)
    cv2.waitKey(0)

    # Step 3: Image Resizing
    width = 600
    height = int((image.shape[0] / image.shape[1]) * width)
    resized_image = cv2.resize(preprocessed_image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow('Resized Image', resized_image)
    cv2.imwrite('resized_image.jpg', resized_image)
    cv2.waitKey(0)

    # Step 4: Grayscale Conversion
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', gray_image)
    gray_image_path = 'gray_image.jpg'
    cv2.imwrite(gray_image_path, gray_image)
    cv2.waitKey(0)

    # Step 5: Edge Detection
    edges = cv2.Canny(gray_image, 50, 150)
    cv2.imshow('Edges', edges)
    cv2.imwrite('edges.jpg', edges)
    cv2.waitKey(0)

    # Step 6: Morphological Transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Morphed Image', morphed_image)
    cv2.imwrite('morphed_image.jpg', morphed_image)
    cv2.waitKey(0)

    # Step 7: Contour Detection
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = resized_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', contour_image)
    cv2.imwrite('contours.jpg', contour_image)
    cv2.waitKey(0)

    # Step 8: License Plate Extraction
    license_plate = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 2 < aspect_ratio < 6 and w > 100 and h > 30:
            license_plate = resized_image[y:y + h, x:x + w]
            break
    if license_plate is not None:
        cv2.imshow('License Plate', license_plate)
        cv2.imwrite('license_plate.jpg', license_plate)
        cv2.waitKey(0)
    else:
        print("Error: Could not detect license plate")
        return None

    # Step 9: Grayscale and Adaptive Thresholding for OCR
    gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    adaptive_thresh_plate = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                  11, 2)
    cv2.imshow('Adaptive Threshold Plate', adaptive_thresh_plate)
    cv2.imwrite('adaptive_thresh_plate.jpg', adaptive_thresh_plate)
    cv2.waitKey(0)

    # Step 10: Character Recognition
    text = pytesseract.image_to_string(adaptive_thresh_plate, config='--psm 8')
    print(f'Recognized License Plate: {text.strip()}')

    # Close all windows
    cv2.destroyAllWindows()
    return text


# Example usage
image_path = 'Case1.png'
recognized_text = process_image(image_path)
if recognized_text:
    print(f'License Plate: {recognized_text}')
