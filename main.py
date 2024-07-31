import cv2
import numpy as np
import os

def process_image(image_path):
    # Step 1: Image Acquisition
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not read image from the given path.")
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    # Step 2: Image Resizing
    width = 600
    height = int((image.shape[0] / image.shape[1]) * width)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow('Resized Image', resized_image)
    cv2.waitKey(0)

    # Step 3: Grayscale Conversion
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', gray_image)
    cv2.waitKey(0)

    # Step 4: Noise Reduction
    preprocessed_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    cv2.imshow('Preprocessed Image', preprocessed_image)
    cv2.waitKey(0)

    # Step 5: Histogram Equalization
    equalized_image = cv2.equalizeHist(preprocessed_image)
    cv2.imshow('Equalized Image', equalized_image)
    cv2.waitKey(0)

    # Step 6: Morphological Transformations
    edges = cv2.Canny(equalized_image, 50, 150)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Morphed Image', morphed_image)
    cv2.waitKey(0)

    return morphed_image, resized_image

def extract_license_plate(morphed_image, resized_image):
    # Step 7: License Plate Extraction
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 2 < aspect_ratio < 6 and w > 100 and h > 30:
            license_plate = resized_image[y:y + h, x:x + w]
            cv2.imshow('License Plate', license_plate)
            cv2.waitKey(0)
            return license_plate
    print("Error: Could not detect license plate")
    return None

def detect_blue_characters(license_plate_image):
    # Step 8: Character Detection
    hsv_image = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_characters = []
    margin = 5

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 100:
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(license_plate_image.shape[1] - x, w + 2 * margin)
            h = min(license_plate_image.shape[0] - y, h + 2 * margin)
            char_img = license_plate_image[y:y + h, x:x + w]
            detected_characters.append((x, y, w, h, char_img))

    avg_width = np.mean([char[2] for char in detected_characters])
    avg_height = np.mean([char[3] for char in detected_characters])
    filtered_characters = [char for char in detected_characters if
                           0.5 * avg_width < char[2] < 1.5 * avg_width and 0.5 * avg_height < char[3] < 1.5 * avg_height]
    filtered_characters.sort(key=lambda c: c[0])

    for i, (x, y, w, h, char_img) in enumerate(filtered_characters):
        cv2.rectangle(license_plate_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow(f'Detected Character {i + 1}', char_img)
        cv2.waitKey(0)

    cv2.imshow('Filtered Detected Characters', license_plate_image)
    cv2.waitKey(0)

    return [char[4] for char in filtered_characters]

def match_characters_with_templates(detected_characters, alphabet_dir, numbers_dir):
    # Step 9: Character Matching
    matched_characters = []
    score_threshold = 0.1
    for idx, char_img in enumerate(detected_characters):
        char_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        char_h, char_w = char_gray.shape[:2]

        best_match = None
        best_score = -np.inf

        if idx < 3:
            template_dir = alphabet_dir
        else:
            template_dir = numbers_dir

        for template_name in os.listdir(template_dir):
            template_path = os.path.join(template_dir, template_name)
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                continue

            template_img_resized = cv2.resize(template_img, (char_w, char_h))
            _, template_img_resized = cv2.threshold(template_img_resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            _, char_gray_thresholded = cv2.threshold(char_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            result = cv2.matchTemplate(char_gray_thresholded, template_img_resized, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            print(f"Character {idx + 1}, Template {template_name}: Score = {score}")
            if score > best_score and score > score_threshold:
                best_score = score
                best_match = template_name.split('.')[0]

        if best_match is not None:
            matched_characters.append(best_match)
            print(f"Detected character: {best_match} with score: {best_score}")
        else:
            matched_characters.append('?')

    return matched_characters

# Example usage
image_path = '01.jpg'
alphabet_dir = './Alphabet'
numbers_dir = './Numbers'

morphed_image, resized_image = process_image(image_path)
license_plate_image = extract_license_plate(morphed_image, resized_image)
if license_plate_image is not None:
    detected_characters = detect_blue_characters(license_plate_image)
    matched_characters = match_characters_with_templates(detected_characters, alphabet_dir, numbers_dir)
    print(f"Matched characters: {''.join(matched_characters)}")
