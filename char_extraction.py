import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def preprocess_image(image_path):
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Resize to standard width
    image = imutils.resize(image, width=500)
    
    # Create a copy for final output
    original_img = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while keeping edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Find edges of the grayscale image
    edged = cv2.Canny(gray, 170, 200)
    
    return original_img, gray, edged

def find_license_plate(original_img, edged):
    """
    Find the license plate contour in the image
    """
    # Find contours based on edges
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]  # Consider top 30 contours
    
    # Initialize license plate contour
    license_plate_cnt = None
    license_plate_roi = None
    
    # Loop over contours to find the best possible approximate contour of the license plate
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If our approximated contour has four points, we can assume it's the license plate
        if len(approx) == 4:
            license_plate_cnt = approx
            x, y, w, h = cv2.boundingRect(c)
            license_plate_roi = original_img[y:y+h, x:x+w]
            break
    
    return original_img, license_plate_cnt, license_plate_roi

def rotate_plate(license_plate_roi, license_plate_cnt):
    """
    Correct rotation in the license plate if needed
    """
    if license_plate_cnt is None or license_plate_roi is None:
        return None
    
    # Function to calculate distance between two points
    def dist(x1, x2, y1, y2):
        return ((x1-x2)**2+(y1-y2)**2)**0.5
    
    # Find bottom coordinates for rotation correction
    idx = 0
    max_y = 0
    
    # Find the index with maximum y-coordinate (bottom-most)
    for i in range(4):
        if license_plate_cnt[i][0][1] > max_y:
            idx = i
            max_y = license_plate_cnt[i][0][1]
    
    # Find indices of adjacent points
    prev_idx = (idx - 1) % 4
    next_idx = (idx + 1) % 4
    
    # Calculate distances to adjacent points
    prev_dist = dist(license_plate_cnt[idx][0][0], license_plate_cnt[prev_idx][0][0], 
                    license_plate_cnt[idx][0][1], license_plate_cnt[prev_idx][0][1])
    
    next_dist = dist(license_plate_cnt[idx][0][0], license_plate_cnt[next_idx][0][0], 
                    license_plate_cnt[idx][0][1], license_plate_cnt[next_idx][0][1])
    
    # Determine left and right bottom points
    if prev_dist > next_dist:
        if license_plate_cnt[prev_idx][0][0] < license_plate_cnt[idx][0][0]:
            left = prev_idx
            right = idx
        else:
            left = idx
            right = prev_idx
    else:
        if license_plate_cnt[next_idx][0][0] < license_plate_cnt[idx][0][0]:
            left = next_idx
            right = idx
        else:
            left = idx
            right = next_idx
    
    # Extract coordinates
    left_x = license_plate_cnt[left][0][0]
    left_y = license_plate_cnt[left][0][1]
    right_x = license_plate_cnt[right][0][0]
    right_y = license_plate_cnt[right][0][1]
    
    # Calculate rotation angle
    opp = right_y - left_y
    hyp = ((left_x-right_x)**2 + (left_y-right_y)**2)**0.5
    
    if hyp == 0:  # Avoid division by zero
        return license_plate_roi
        
    sin = opp / hyp
    theta = math.asin(sin) * 57.2958  # Convert to degrees
    
    # Rotate the image
    image_center = tuple(np.array(license_plate_roi.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, theta, 1.0)
    result = cv2.warpAffine(license_plate_roi, rot_mat, license_plate_roi.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    # Crop rotated image appropriately
    if opp > 0:
        h = result.shape[0] - opp // 2
    else:
        h = result.shape[0] + opp // 2
    
    result = result[0:h, :]
    return result

def segment_characters(plate_img):
    
    if plate_img is None:
        return []
    
    # Resize the license plate for better character segmentation
    img_lp = cv2.resize(plate_img, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Apply morphological operations (erosion and dilation)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))
    
    # Make borders white
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]
    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255
    
    # Define character dimensions
    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]
    
    # Find character contours
    return find_contours(dimensions, img_binary_lp)

def find_contours(dimensions, img):
    """
    Find character contours in the license plate
    """
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Get top 15 contours
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    
    for cntr in cntrs:
        # Get bounding box
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # Filter by dimensions
        if (intWidth > lower_width and intWidth < upper_width and 
            intHeight > lower_height and intHeight < upper_height):
            
            x_cntr_list.append(intX)  # Store x-coordinate for sorting
            
            # Prepare character image (44x24)
            char_copy = np.zeros((44, 24))
            
            # Extract character
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            # Invert colors (background = black, text = white)
            char = cv2.subtract(255, char)
            
            # Place character in the center of the fixed-size image
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0
            
            img_res.append(char_copy)
    
    # Sort characters from left to right
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    
    return np.array(img_res_copy)

def create_model():
    """
    Create CNN model with the same architecture as the trained model
    """
    model = Sequential()
    model.add(Conv2D(16, (22, 22), input_shape=(28, 28, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (16, 16), input_shape=(28, 28, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (8, 8), input_shape=(28, 28, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (4, 4), input_shape=(28, 28, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(36, activation='softmax'))
    
    return model

def fix_dimension(img):
    """
    Convert grayscale image to 3-channel image for model prediction
    """
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img

def recognize_plate(char_list, model):
    """
    Recognize license plate characters using the pre-trained model
    """
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    dic = {i: c for i, c in enumerate(characters)}
    
    output = []
    for char in char_list:
        img = cv2.resize(char, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img)
        img = img.reshape(1, 28, 28, 3)  # Prepare image for model
        
        y_pred = np.argmax(model.predict(img, verbose=0), axis=-1)[0]
        character = dic[y_pred]
        output.append(character)
    
    plate_number = ''.join(output)
    return plate_number

def process_license_plate_image(image_path, weights_path):
    """
    Process a new license plate image and extract the text
    """
    # Preprocess the image
    original_img, gray, edged = preprocess_image(image_path)
    
    # Find license plate
    img_with_contour, license_plate_cnt, license_plate_roi = find_license_plate(original_img, edged)
    
    if license_plate_roi is None:
        return "No license plate detected", None, None, None
    
    # Rotate license plate if needed
    rotated_plate = rotate_plate(license_plate_roi, license_plate_cnt)
    
    if rotated_plate is None:
        return "Failed to process license plate", img_with_contour, license_plate_roi, None
    
    # Segment characters
    char_list = segment_characters(rotated_plate)
    
    if len(char_list) == 0:
        return "No characters detected in license plate", img_with_contour, license_plate_roi, rotated_plate
    
    # Create and load the model
    model = create_model()
    try:
        model.load_weights(weights_path)
    except Exception as e:
        return f"Failed to load model weights: {str(e)}", img_with_contour, license_plate_roi, rotated_plate
    
    # Recognize license plate
    plate_number = recognize_plate(char_list, model)
    
    return plate_number, img_with_contour, license_plate_roi, rotated_plate

def visualize_results(plate_number, img_with_contour, license_plate_roi, rotated_plate, char_list=None):
    """
    Visualize the results
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img_with_contour, cv2.COLOR_BGR2RGB))
    plt.title("Detected License Plate")
    plt.axis('off')
    
    if license_plate_roi is not None:
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(license_plate_roi, cv2.COLOR_BGR2RGB))
        plt.title("Extracted License Plate")
        plt.axis('off')
    
    if rotated_plate is not None:
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(rotated_plate, cv2.COLOR_BGR2RGB))
        plt.title("Rotated License Plate")
        plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, f"Recognized Text: {plate_number}", 
             horizontalalignment='center', verticalalignment='center', fontsize=15)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display segmented characters if available
    if char_list is not None and len(char_list) > 0:
        plt.figure(figsize=(12, 4))
        for i, char in enumerate(char_list):
            if i < 12:  # Display up to 12 characters
                plt.subplot(1, min(len(char_list), 12), i+1)
                plt.imshow(char, cmap='gray')
                plt.axis('off')
        plt.suptitle("Segmented Characters")
        plt.tight_layout()
        plt.show()

def main(image_path, weights_path):
    # Set the paths to your new image and saved model weights
    image_path = r"C:\Users\Lenovo\DEVELOPERS SECTION\License-Plate-Number-Detection-main\Artificial Mercosur License Plates\train\default\cropped_parking_lot_164.jpg"
    weights_path = r"C:\Users\Lenovo\DEVELOPERS SECTION\License-Plate-Number-Detection-main\checkpoints\my_checkpoint.weights.h5"

    
    # Process the license plate image
    plate_number, img_with_contour, license_plate_roi, rotated_plate = process_license_plate_image(image_path, weights_path)
    
    # If plate was detected, segment characters for visualization
    char_list = None
    if rotated_plate is not None:
        char_list = segment_characters(rotated_plate)
    
    # Visualize the results
    visualize_results(plate_number, img_with_contour, license_plate_roi, rotated_plate, char_list)
    
    return plate_number

if __name__ == "__main__":
    image_path = r"C:\Users\Lenovo\DEVELOPERS SECTION\License-Plate-Number-Detection-main\Artificial Mercosur License Plates\train\default\cropped_parking_lot_164.jpg"
    weights_path = r"C:\Users\Lenovo\DEVELOPERS SECTION\License-Plate-Number-Detection-main\checkpoints\my_checkpoint.weights.h5"

    result = main(image_path, weights_path)
    print("Recognized license plate:", result)