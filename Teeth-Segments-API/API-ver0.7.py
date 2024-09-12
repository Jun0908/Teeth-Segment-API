import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow.keras.models import load_model
import csv
from PIL import Image
from io import BytesIO
import base64
from scipy.spatial import distance as dist
from imutils import perspective
import pandas as pd
import uuid

app = Flask(__name__)
CORS(app)

# Prepare the model and data
model_path = 'model/unet_model.h5'
model = load_model(model_path)

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello'})

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file name'}), 400
    
    try:
        # Load the image and save it in PNG format
        image = Image.open(file.stream)
        image.save('sample.png', 'PNG')
        return jsonify({'message': 'Image successfully uploaded'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to convert an image to a Base64-encoded string
def img_to_base64_str(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def preprocess_image(image_path, target_size=(512, 512)):
    img = Image.open(image_path)
    img = img.resize(target_size, Image.ANTIALIAS)
    if img.mode != 'L':
        img = img.convert('L')
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# Function to save the prediction result as an image
def save_prediction(prediction, save_path='predict.png'):  # Specify default value for save path
    prediction_image = (prediction.squeeze() * 255).astype(np.uint8)
    img = Image.fromarray(prediction_image)
    img.save(save_path)
    print(f"Prediction saved to {save_path}")


def apply_color_based_on_intensity(input_path, output_path):
    image_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    high_intensity_threshold = 200  # Above this, apply yellow
    low_intensity_threshold = 50    # Below this, apply purple
    image_rgb[image_gray > high_intensity_threshold] = [255, 255, 0]  # Yellow
    image_rgb[image_gray < low_intensity_threshold] = [128, 0, 128]  # Purple
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

@app.route('/predict', methods=['GET'])
def predict():
    # Execute prediction and return the result as a Base64 string
    sample_image_path = 'sample.png'
    preprocessed_img = preprocess_image(sample_image_path)
    predict_img = model.predict(preprocessed_img)
    save_prediction(predict_img)  # This function saves the prediction result in 'predict.png'
    # Call the apply_color_based_on_intensity function by passing file paths directly
    apply_color_based_on_intensity('predict.png', 'modified_predict.png')  # Directly specify file paths
    image = cv2.imread("./modified_predict.png")
    base64_str = img_to_base64_str(image)
    return jsonify({"image_base64": base64_str})

@app.route('/contours', methods=['GET'])
def contours():
    # Extract contours and return the result as a Base64 string
    image = cv2.imread('modified_predict.png')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_only_image = np.zeros_like(image_rgb)
    cv2.drawContours(contour_only_image, contours, -1, (0, 255, 0), 2)
    base64_str = img_to_base64_str(cv2.cvtColor(contour_only_image, cv2.COLOR_RGB2BGR))
    return jsonify({"image_base64": base64_str})

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def write_to_csv(data, file_name):
    with open(file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def save_segmentation_dimensions(contours, csv_file_name='segmentation_dimensions.csv'):
    for i, cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")

        # Calculate the dimensions of the rectangle
        width = dist.euclidean(box[0], box[1])
        height = dist.euclidean(box[0], box[3])

        # Data to write to CSV
        data = [i + 1, width, height] + box.flatten().tolist()
        write_to_csv(data, csv_file_name)

def process_image(original_image, contours, csv_file_name='midpoints.csv'):
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:  # Set minimum size for area
            count += 1
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            color = (list(np.random.choice(range(256), size=3)))

            # Calculate midpoints
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # Write midpoints to CSV
            write_to_csv([count, tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY], csv_file_name)
            
            color = (list(np.random.choice(range(256), size=3)))  
            color = [int(color[0]), int(color[1]), int(color[2])]

            # Draw enclosures and midpoints on the image
            cv2.drawContours(original_image, [box.astype("int")], 0, color, 2)
            cv2.circle(original_image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(original_image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(original_image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(original_image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            # Calculate dimensions
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # Display dimensions on the image
            cv2.putText(original_image, "{:.1f}px".format(dA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(original_image, "{:.1f}px".format(dB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return original_image, count

def sort_and_replace(csv_input_filename, final_output_filename):
    # Load data from the CSV file
    with open(csv_input_filename, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Add an index to each row for sorting based on the X and Y coordinates of tltr
    for i, row in enumerate(rows):
        x_coord = float(row[1])  # X coordinate of tltr
        y_coord = float(row[2])  # Y coordinate of tltr
        rows[i] = (i, x_coord, y_coord, row)  # Index, X coordinate, Y coordinate, original row

    # Group by Y coordinate (upper and lower teeth)
    rows.sort(key=lambda x: x[2])  # Sort by Y coordinate
    median_y = rows[len(rows) // 2][2]  # Median Y coordinate

    # Split into lower and upper teeth
    lower_teeth = [row for row in rows if row[2] > median_y]
    upper_teeth = [row for row in rows if row[2] <= median_y]

    # Sort lower teeth by X coordinate
    lower_teeth.sort(key=lambda x: x[1])
    # Sort upper teeth by X coordinate
    upper_teeth.sort(key=lambda x: x[1])

    # Get sorted rows and replace the first column value with row number
    sorted_rows = [row[3] for row in lower_teeth + upper_teeth]
    for i, row in enumerate(sorted_rows):
        row_number = str(i + 1)  # Row number is index + 1
        row[0] = row_number

    # Write the replaced data to the final file
    with open(final_output_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sorted_rows)

def load_and_prepare_data(file_path):
    """Load data from a CSV file and calculate the median Y coordinate"""
    df = pd.read_csv(file_path)
    y_coords_upper = df.iloc[:, 2]  # Y coordinates of upper teeth (column C)
    y_coords_lower = df.iloc[:, 4]  # Y coordinates of lower teeth (column E)
    y_coords_all = pd.concat([y_coords_upper, y_coords_lower])
    median_y = np.median(y_coords_all)
    return df, median_y

def process_image_and_save_points(df, median_y, image_path, output_file):
    """Plot dots on the image, save plotted points to a CSV, and return the converted RGB image"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plotted_points = []
    for index, row in df.iterrows():
        if row[2] <= median_y:  # For upper teeth
            cv2.circle(image_rgb, (int(row[1]), int(row[2])), 5, (255, 0, 0), -1)
            plotted_points.append((int(row[1]), int(row[2])))
        else:  # For lower teeth
            cv2.circle(image_rgb, (int(row[3]), int(row[4])), 5, (255, 0, 0), -1)
            plotted_points.append((int(row[3]), int(row[4])))
    # Save plotted coordinates to a CSV file
    #plotted_df = pd.DataFrame(plotted_points, columns=['X', 'Y'])
    # funabashi 小文字に変更
    plotted_df = pd.DataFrame(plotted_points, columns=['x', 'y'])
    plotted_df.to_csv(output_file, index=False)
    return image_rgb

def read_midpoints_from_csv(csv_file_name):
    midpoints = []
    with open(csv_file_name, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row:  # Ignore empty lines
                trbr = (float(row[7]), float(row[8]))  # Right side midpoint
                tlbl = (float(row[5]), float(row[6]))  # Left side midpoint
                midpoints.append((trbr, tlbl))
    return midpoints

def calculate_adjacent_midpoints(midpoints):
    adjacent_midpoints = []
    for i in range(len(midpoints) - 1):
        midpoint1 = midpoints[i][0]   # Right side midpoint of the i-th contour
        midpoint2 = midpoints[i+1][1]  # Left side midpoint of the i+1-th contour
        adjacent_midpoint = ((midpoint1[0] + midpoint2[0]) / 2, (midpoint1[1] + midpoint2[1]) / 2)
        adjacent_midpoints.append(adjacent_midpoint)
    return adjacent_midpoints

def write_adjacent_midpoints_to_csv(adjacent_midpoints, csv_file_name):
    with open(csv_file_name, 'a', newline='') as f:  # Change 'w' to 'a' to append at the end of the file
        writer = csv.writer(f)
        for midpoint in adjacent_midpoints:
            writer.writerow(midpoint)

def draw_and_return_midpoints_image(image_path, midpoints, output_image_path):
    image = cv2.imread(image_path)
    for midpoint in midpoints:
        cv2.circle(image, (int(midpoint[0]), int(midpoint[1])), 5, (0, 255, 0), -1)
    cv2.imwrite(output_image_path, image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb  # This function returns the modified image's RGB array

@app.route('/processed_imaged', methods=['GET'])
def get_processed_image():
    # Image paths
    original_image_path = './sample.png'
    mask_image_path = 'modified_predict.png'
    
    original_image = cv2.imread(original_image_path)
    mask_image = cv2.imread(mask_image_path)

    # Check the size of the original and mask images
    if original_image.shape[:2] != mask_image.shape[:2]:
    # If sizes differ, resize the mask image
     mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]))

    # Convert mask image to RGB format
    mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(mask_image_rgb, cv2.COLOR_RGB2HSV)

    # Define the range for yellow
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # Create a mask for yellow
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Detect contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    save_segmentation_dimensions(contours, 'segmentation_dimensions.csv')
    processed_image, segment_count = process_image(original_image, contours, 'midpoints.csv')

    retval, buffer = cv2.imencode('.jpg', processed_image)
    img_str = base64.b64encode(buffer).decode()

    return jsonify(image_base64=img_str)


@app.route('/download_csv/midpoints', methods=['GET'])
def get_csv_data():
    file_path = 'midpoints.csv'
    data = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    
@app.route('/histograms', methods=['GET'])
def generate_histograms():
    image_path = './predict.png'
    csv_path = './histograms.csv'

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Detect contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if the CSV file exists (create if it doesn't)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['ContourID', 'HistogramData'])  # Add header row

    # Write histogram data to the CSV file
    with open(csv_path, 'a', newline='') as csvfile:  # Use 'a' for append mode instead of 'w'
        csv_writer = csv.writer(csvfile)

        for i, contour in enumerate(contours):
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            pixels = image[mask == 255]
            hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
            csv_writer.writerow(['Tooth_' + str(i)] + list(hist))

    # Load histogram data from the CSV file
    histograms = []
    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            histograms.append([int(value) for value in row[1:]])

    return jsonify(histograms)

@app.route('/download_csv/final_plotted_points', methods=['GET'])
def process_data():
    #image_path = 'sample.png'
    arranged_csv = 'final_output.csv'
    csv_file_name = 'midpoints.csv'
    output_csv_file = 'final_plotted_points.csv'
    original_image_path = 'sample.png'
    output_image_path = 'output_image_with_midpoints.png'

    # Implementation of these functions is omitted
    sort_and_replace(csv_file_name, arranged_csv)
    df, median_y = load_and_prepare_data(csv_file_name)
    image_rgb_1 = process_image_and_save_points(df, median_y, original_image_path, output_csv_file)
    cv2.imwrite('image_rgb_1.png', image_rgb_1)
    midpoints_data = read_midpoints_from_csv(arranged_csv)
    adjacent_midpoints = calculate_adjacent_midpoints(midpoints_data)
    write_adjacent_midpoints_to_csv(adjacent_midpoints, output_csv_file)
    image_rgb_2 = draw_and_return_midpoints_image(original_image_path, adjacent_midpoints, output_image_path)
    cv2.imwrite('image_rgb_2.png', image_rgb_2)
    data = []

    try:
        with open(output_csv_file, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route('/file-to-csv', methods=['POST'])
def upload_image2():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file name'}), 400
    
    try:
        # Load the image and save it in PNG format
        image = Image.open(file.stream)
        base_name = str(uuid.uuid1());
        print('base_name : ' + base_name);

        input_image_path = base_name + '.png'
        
        image.save(input_image_path, 'PNG')


        #return jsonify({'message': 'Image successfully uploaded'}), 200


        # Execute prediction and return the result as a Base64 string

        print('start predict');
        preprocessed_img = preprocess_image(input_image_path)
        predict_img = model.predict(preprocessed_img)

        predict_img_path = base_name + '_predict.png';
        modified_predict_img_path = base_name + '_modified_predict.png';
        
        save_prediction(predict_img, predict_img_path)  # This function saves the prediction result in 'predict.png'
        # Call the apply_color_based_on_intensity function by passing file paths directly
    
        apply_color_based_on_intensity(predict_img_path, modified_predict_img_path)  # Directly specify file paths
        image = cv2.imread(modified_predict_img_path)
        base64_str = img_to_base64_str(image)

        #return jsonify({"image_base64": base64_str})
        
        # Extract contours and return the result as a Base64 string

        print('start contours');
        
        image = cv2.imread(modified_predict_img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_only_image = np.zeros_like(image_rgb)
        cv2.drawContours(contour_only_image, contours, -1, (0, 255, 0), 2)
        base64_str = img_to_base64_str(cv2.cvtColor(contour_only_image, cv2.COLOR_RGB2BGR))
        #return jsonify({"image_base64": base64_str})

        print('start processed_imaged');
        
        # Image paths
        original_image = cv2.imread(input_image_path)
        mask_image = cv2.imread(modified_predict_img_path)

        # Check the size of the original and mask images
        if original_image.shape[:2] != mask_image.shape[:2]:
        # If sizes differ, resize the mask image
         mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]))

        # Convert mask image to RGB format
        mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

        # Convert to HSV color space
        hsv_image = cv2.cvtColor(mask_image_rgb, cv2.COLOR_RGB2HSV)

        # Define the range for yellow
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

        # Create a mask for yellow
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # Detect contours from the mask
        segmentation_dimensions_csv_path = base_name + '_segmentation_dimensions.csv';
        midpoints_csv_path = base_name + '_midpoints.csv';
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        save_segmentation_dimensions(contours, segmentation_dimensions_csv_path)
        processed_image, segment_count = process_image(original_image, contours, midpoints_csv_path)

        retval, buffer = cv2.imencode('.jpg', processed_image)
        img_str = base64.b64encode(buffer).decode()

        #return jsonify(image_base64=img_str)

        print('start process_data');

        arranged_csv = base_name + '_final_output.csv'
        output_csv_file = base_name + 'final_plotted_points.csv'        
        output_image_path = base_name + 'output_image_with_midpoints.png'

        # Implementation of these functions is omitted
        sort_and_replace(midpoints_csv_path, arranged_csv)
        df, median_y = load_and_prepare_data(midpoints_csv_path)
        image_rgb_1 = process_image_and_save_points(df, median_y, input_image_path, output_csv_file)
        cv2.imwrite(base_name + '_image_rgb_1.png', image_rgb_1)
        midpoints_data = read_midpoints_from_csv(arranged_csv)
        adjacent_midpoints = calculate_adjacent_midpoints(midpoints_data)
        write_adjacent_midpoints_to_csv(adjacent_midpoints, output_csv_file)
        image_rgb_2 = draw_and_return_midpoints_image(input_image_path, adjacent_midpoints, output_image_path)
        cv2.imwrite(base_name + '_image_rgb_2.png', image_rgb_2)
        data = []

        try:
            with open(output_csv_file, mode='r', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    data.append(row)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8000)

