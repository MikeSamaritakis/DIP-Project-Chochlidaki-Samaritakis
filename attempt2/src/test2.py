from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import csv


# Image Processing functions

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)  # Edge detection
    
    # Morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return blurred, gray, edges, morph

def process_image_with_morphological_operations(image_path):
    image = cv2.imread(image_path)    
    # Preprocess the image
    blurred, gray, edges, morph = process_image(image)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_stitches = len(contours)  # Count the number of contours as the number of stitches
    
    return n_stitches, gray, blurred, edges, morph


# Image Quality Evaluation function

def evaluate_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Line detection using Hough Line Transform
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=40)
    if lines is None:
        return -1

    return 0


# Visualization functions

def visualize_processing_bad_data(gray, edges, filename, output_folder):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    axs[0].imshow(gray, cmap='gray')
    axs[0].set_title('Grayscale Image')
    axs[0].axis('off')
    
    axs[1].imshow(edges, cmap='gray')
    axs[1].set_title('Edges Detected')
    axs[1].axis('off')

    fig.suptitle('Test2 Number of Stitches: -1', fontsize=15)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{filename}.png")
    plt.savefig(output_path)
    plt.close()

def visualize_processing(n_stitches, gray, blurred, edges, morph, filename, output_folder):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    axs[0, 0].imshow(gray, cmap='gray')
    axs[0, 0].set_title('Grayscale Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(blurred, cmap='gray')
    axs[0, 1].set_title('Blurred Image')
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(edges, cmap='gray')
    axs[1, 0].set_title('Edges Detected')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(morph, cmap='gray')
    axs[1, 1].set_title('Morphological Operations Result')
    axs[1, 1].axis('off')

    fig.suptitle(f'Test2 Number of Stitches: {n_stitches}', fontsize=15)
    
    plt.tight_layout()
    output_path = os.path.join(output_folder, f"{filename}.png")
    plt.savefig(output_path)
    plt.close()


# Main function

if __name__ == "__main__":
    directory_path = 'test_images'
    output_folder = 'output_images_from_test2'
    bad_data_folder = 'output_images_with_bad_data_from_test2'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(bad_data_folder, exist_ok=True)
    
    csv_file = 'results_test2.csv'
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
            check = evaluate_image_quality(image)
            if check == -1:
                gray, edges = process_image(image)[1:3]
                visualize_processing_bad_data(gray, edges, filename, bad_data_folder)
                writer.writerow([filename, -1])
                continue
            else:
                n_stitches, gray, blurred, edges, morph = process_image_with_morphological_operations(image_path)
                results[filename] = n_stitches
                writer.writerow([filename, n_stitches])
                
                visualize_processing(n_stitches, gray, blurred, edges, morph, filename, output_folder)

    print(f"Results saved to {csv_file}")
