from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from math import sqrt, atan2, degrees
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import xmltodict
import argparse
import json
import csv
import sys
import json
import cv2
import math
import os

xml_path = 'resources/annotations.xml'

# Given code starts here

with open(xml_path) as fd:
    doc = xmltodict.parse(fd.read())

doc["annotations"].keys()

doc["annotations"]["meta"]["task"]["updated"]

doc["annotations"]["image"][0]

d = {
     "asas": 15,
     "sdf": [[1.2], [5] ]
}

data = [
    { "filename": "incision001.jpg",
      "incision_polyline": [[ 109.47, 19.32],[111.88,42.19]],
      "crossing_positions": [13.8, 18.1, 19.0],
      "crossing_angles": [87.1, 92.3, 75.0],
    },
  ]

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

def intersectLines( pt1, pt2, ptA, ptB ):
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;   x2, y2 = pt2
    dx1 = x2 - x1;  dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;   xB, yB = ptB;
    dx = xB - x;  dy = yB - y;

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

    # now, the determinant should be OK
    DETinv = 1.0/DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

    # return the average of the two descriptions
    xi = (x1 + r*dx1 + x + s*dx)/2.0
    yi = (y1 + r*dy1 + y + s*dy)/2.0

    #found is intersection (xi,yi) in inner segment
    valid = 0
    if x1 != x2:
        if x1 < x2:
            a = x1
            b = x2
        else:
            a = x2
            b = x1
        c = xi
    else:
        #predpoklad, ze pak y jsou ruzne
        if y1 < y2:
            a = y1
            b = y2
        else:
            a = y2
            b = y1
        c = yi
    if (c > a) and (c < b):
        #now second segment
        if x != xB:
            if x < xB:
                a = x
                b = xB
            else:
                a = xB
                b = x
            c = xi
        else:
            #predpoklad, ze pak y jsou ruzne
            if y < yB:
                a = y
                b = yB
            else:
                a = yB
                b = y
            c = yi
        if (c > a) and (c < b):
            valid = 1

    return ( xi, yi, valid, r, s )

def proces_data(data_path):
    with open(data_path+'/annotations.json') as json_file:
        data = json.load(json_file)

    annotations = {}
    for d in data:
        file_name = data_path+d['name']
        print(file_name, end=' ')
        im = cv2.imread(file_name)
        incisions = np.array(d['incisions'], dtype=object)
        for p_i in incisions:
            p = np.array(p_i)
            plt.plot(p[:,0], p[:,1])
        
        stitches = np.array(d['stitches'], dtype=object)

        # Editing given code

        if len(incisions) == 0:
            print(f", -1 # image could not be processed") # bad data
        else:
            print(f", {len(stitches)} # image contains {len(stitches)} stiches") # print the number of stitches

        # Editing given code ends here

        for p_s in stitches:
            p = np.array(p_s)
            plt.plot(p[:,0], p[:,1])

        incision_alphas = []
        incision_lines = []
        for incision in incisions:
            for (p_1, p_2) in zip(incision[:-1],incision[1:]):
                p1 = np.array(p_1)
                p2 = np.array(p_2)
                dx = p2[0]-p1[0]
                dy = p2[1]-p1[1]
                if dy == 0:
                    alpha = 90.0
                elif dx == 0:
                    alpha = 0.0
                else:
                    alpha = 90 + 180.*np.arctan(dy/dx)/np.pi
                incision_alphas.append(alpha)
                incision_lines.append([p1, p2])

        stitche_alphas = []
        stitche_lines = []
        for stitche in stitches:
            for (p_1, p_2) in zip(stitche[:-1],stitche[1:]):
                p1 = np.array(p_1)
                p2 = np.array(p_2)
                dx = p2[0]-p1[0]
                dy = p2[1]-p1[1]
                if dy == 0:
                    alpha = 90.0
                elif dx == 0:
                    alpha = 180.0
                else:
                    alpha = 90 + 180.*np.arctan(dy/dx)/np.pi
                stitche_alphas.append(alpha)
                stitche_lines.append([p1, p2])

        # analyze alpha for each pair of line segments
        intersections = []
        intersections_alphas = []
        for (incision_line, incision_alpha) in zip(incision_lines, incision_alphas):
            for (stitche_line, stitche_alpha) in zip(stitche_lines, stitche_alphas):

                p0, p1 = incision_line
                pA, pB = stitche_line
                (xi, yi, valid, r, s) = intersectLines(p0, p1, pA, pB)
                if valid == 1:
                    intersections.append([xi, yi])
                    alpha_diff = abs(incision_alpha - stitche_alpha)
                    alpha_diff = 180.0 - alpha_diff if alpha_diff > 90.0 else alpha_diff
                    alpha_diff = 90 - alpha_diff
                    intersections_alphas.append(alpha_diff)

        # visualize
        if True:
            plt.imshow(im)
            for ((xi,yi), alpha) in zip(intersections, intersections_alphas):

                plt.plot(xi, yi, 'o')
                plt.text(xi, yi,'{:2.1f}'.format(alpha), c='green', bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 1}, size='large')

            #plt.show()

    return intersections, intersections_alphas

# Given code ends here

def display_results(image, count): 
    # Display the image with the count
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"Count: {count}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur (L%P)
    return blurred

def apply_gaussian_blur(gray_image, kernel_size=(5, 5), sigma=0):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, kernel_size, sigma)
    return blurred_image

def write_to_csv(results, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'n_stitches'])
        writer.writerows(results)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Stitch Counter')
    parser.add_argument('output_csv', type=str, help='The CSV file to output results')
    parser.add_argument('-v', '--visual', action='store_true', help='Visual mode to display images')
    parser.add_argument('images', type=str, nargs='+', help='Image filenames')
    parser.add_argument('-m', '--method', choices=['hough', 'predict'], default='hough', help='Method to use for counting stitches: hough or predict')
    return parser.parse_args()

def segment_image(gray_image):
    edges = cv2.Canny(gray_image, 50, 150)  # Edge detection
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    return contours

def visualize_contours(image, contours):
    vis_image = image.copy()
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 1)
    cv2.imshow('Contours', vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def build_model():
    model = Sequential([
        Input(shape=(256, 256, 1)),  # Define the input shape explicitly here
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)  # Assuming a regression output
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and compile the model
model = build_model()

def evaluate_image_quality(image):
    # Check for blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 63:
        return -1
    # Check for brightness
    brightness = np.mean(gray)
    if brightness < 50 or brightness > 200:
        return -1
    # Check for lines using Hough Transform
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=40)
    if lines is None:
        return -1
    # If all checks are passed, return 0 indicating image is OK
    return 0

# Function to calculate length of a segment
def calculate_length(p1, p2):
    return sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Function to calculate angle of a segment
def calculate_angle(p1, p2):
    return degrees(atan2(p2[1] - p1[1], p2[0] - p1[0]))

# Load annotations
def load_annotations():
    with open('resources/annotations.json') as json_file:
        annotations = json.load(json_file)
    return annotations

def extract_features_and_labels(data):
    features = []
    labels = []
    for entry in data:
        lengths = []
        angles = []
        for segment in entry['stitches']:
            for i in range(len(segment) - 1):
                p1, p2 = segment[i], segment[i+1]
                length = calculate_length(p1, p2)
                angle = calculate_angle(p1, p2)
                lengths.append(length)
                angles.append(angle)
        
        # Aggregate features for uniformity
        if lengths:
            feature_vector = [
                np.mean(lengths), np.std(lengths), np.min(lengths), np.max(lengths),
                np.mean(angles), np.std(angles), np.min(angles), np.max(angles)
            ]
        else:
            # Default values if no stitches are detected
            feature_vector = [0] * 8
        features.append(feature_vector)
        labels.append(len(entry['stitches']))  # Number of stitches as label
    
    return np.array(features), np.array(labels)

def train_classifier(features, labels):
    # Perform feature scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform oversampling to handle class imbalance
    oversampler = RandomOverSampler()
    oversampled_features, oversampled_labels = oversampler.fit_resample(scaled_features, labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(oversampled_features, oversampled_labels, test_size=0.2, random_state=42)

    # Create the SVM classifier
    svm = SVC()

    # Define the hyperparameters for grid search
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best classifier from grid search
    best_svm = grid_search.best_estimator_

    # Train the best classifier on the entire training set
    best_svm.fit(X_train, y_train)

    # Evaluate the classifier on the testing set
    y_pred = best_svm.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return best_svm

def segment_image_with_hough_transform(image):
    # Check if the image is already in grayscale
    if len(image.shape) == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image   
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=28, maxLineGap=4)
    # Filter lines based on length and position
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > 10:  
                filtered_lines.append(line[0])
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image, filtered_lines

def process_image_with_hough_transform(image_path):
    image = cv2.imread(image_path)    
    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Segment the image using Hough Transform
    hough_image, lines = segment_image_with_hough_transform(blurred)
    # Count the number of stitches
    if lines is not None:
        n_stitches = len(lines)
    else:
        n_stitches = 0
    return n_stitches, hough_image

if __name__ == "__main__":

    args = parse_arguments()
    image_filenames = args.images
    annotations = load_annotations()
    data_path = 'resources/'  
    features, labels = extract_features_and_labels(annotations)

    print("Number of stitches from annotations:")
    intersections, intersections_alphas = proces_data(data_path)    
    
    #print("Intersections:", intersections)
    #print("Intersection Alphas:", intersections_alphas)

    if len(set(labels)) < 2:
        print("Error: Not enough class diversity for training. Please ensure the dataset contains multiple classes.")
    else:
        classifier = train_classifier(features, labels)
    
    results= []
    for filename in image_filenames:
        image_path = os.path.join('resources/test_images', filename)
        image = cv2.imread(image_path)
        check = evaluate_image_quality(image)

        if image is None:
            print(f"Error: Cannot read image {filename}.")
            results.append((filename, -1))
            continue
        else:
            if check == -1:
                print(f"Error: Image {filename} could not be processed")
                results.append((filename, -1))
                continue
            else:
                if args.method == 'hough':
                    # Preprocess the image
                    preprocessed_image = preprocess_image(image)
                    # Segment the image
                    contours = segment_image(preprocessed_image)
                    # Count the number of stitches using Hough Transform
                    n_stitches, hough_image = process_image_with_hough_transform(image_path)
                else:
                    # Preprocess the image
                    preprocessed_image = preprocess_image(image)
                    # Segment the image
                    contours = segment_image(preprocessed_image)
                    # Predict the labels using the classifier
                    predicted_labels = classifier.predict(features)
                    # Count the number of stitches
                    n_stitches = len(predicted_labels)

                if args.visual:
                    visualize_contours(preprocessed_image, contours)
                    preprocessed_image_path = os.path.join('root', 'report_images', filename)
                    contours_path = os.path.join('root', 'report_images', f"{filename}_contours.jpg")
                    # Save the preprocessed image
                    cv2.imwrite(preprocessed_image_path, preprocessed_image)
                    # Save the image with contours
                    vis_image = image.copy()
                    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 1)
                    cv2.imwrite(contours_path, vis_image)

            results.append((filename, n_stitches))
           
    write_to_csv(results, args.output_csv)