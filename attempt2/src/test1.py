from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import csv


# Image Processing function

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Apply Otsu's thresholding

    # Morphological operations to remove noise and small objects
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return gray, threshold, morph, contours


# Image Quality Evaluation function

def evaluate_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Line detection using Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=40)
    if lines is None:
        return -1, gray, edges, None

    return 0, gray, edges, lines


# Visualization functions

def visualize_processing_bad_data(gray, edges, filename, output_folder):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    axs[0].imshow(gray, cmap='gray')
    axs[0].set_title('Grayscale Image')
    axs[0].axis('off')
    
    axs[1].imshow(edges, cmap='gray')
    axs[1].set_title('Edges Detected')
    axs[1].axis('off')

    fig.suptitle('Test1 Number of Stitches: -1', fontsize=15)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{filename}.png")
    plt.savefig(output_path)
    plt.close()

def visualize_processing(n_stitches, gray, threshold, morph, contour_image, filename, output_folder):
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))
    
    axs[0].imshow(gray, cmap='gray')
    axs[0].set_title('Grayscale Image')
    axs[0].axis('off')
    
    axs[1].imshow(threshold, cmap='gray')
    axs[1].set_title('Otsu Threshold Image')
    axs[1].axis('off')
    
    axs[2].imshow(morph, cmap='gray')
    axs[2].set_title('Morphological Operations')
    axs[2].axis('off')

    axs[3].imshow(contour_image, cmap='gray')
    axs[3].set_title('Contours')
    axs[3].axis('off')
    
    fig.suptitle(f'Test1 Number of Stitches: {n_stitches}', fontsize=15)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{filename}.png")
    plt.savefig(output_path)
    plt.close()


# Main function

if __name__ == "__main__":
    directory_path = 'test_images'
    output_folder = 'output_images_from_test1'
    bad_data_folder = 'output_images_with_bad_data_from_test1'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(bad_data_folder, exist_ok=True)
    
    csv_file = 'results_test1.csv'
    results = {}

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Stitches"])
        
        for filename in os.listdir(directory_path):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Cannot read image {filename}.")
                continue
            check, gray, edges, lines = evaluate_image_quality(image)
            if check == -1:
                visualize_processing_bad_data(gray, edges, filename, bad_data_folder)
                writer.writerow([filename, -1])
                continue
            else:
                gray, threshold, morph, contours = process_image(image)
                n_stitches = len(contours)
                results[filename] = n_stitches
                writer.writerow([filename, n_stitches])
                
                contour_image = np.copy(image)
                cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

                visualize_processing(n_stitches, gray, threshold, morph, contour_image, filename, output_folder)

    print(f"Results saved to {csv_file}")
