## !IGNORE THIS FILE! ##

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_contour_areas(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    return areas

def plot_contour_areas_histogram(areas):
    plt.hist(areas, bins=50)
    plt.xlabel('Contour Area')
    plt.ylabel('Frequency')
    plt.title('Histogram of Contour Areas')
    plt.show()

# Example usage with a directory of sample images
directory_path = 'test_images'
all_areas = []
for filename in os.listdir(directory_path):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(directory_path, filename)
        areas = extract_contour_areas(image_path)
        all_areas.extend(areas)

plot_contour_areas_histogram(all_areas)
