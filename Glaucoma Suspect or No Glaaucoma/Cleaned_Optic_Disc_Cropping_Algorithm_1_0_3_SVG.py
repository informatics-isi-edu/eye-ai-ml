# # cleaned 1.0.3 with bounding box output as well

import os
import cv2 as cv
import cv2
import numpy as np
import pandas as pd
import time
import tensorflow as tf

IMG_SIZE = (300, 300)

preprocess_input = tf.keras.applications.vgg19.preprocess_input

def save_svg(output_directory, annotation_tag_rid, rid, raw_image_size, bbox, annotation_tag_name):
    # Set the viewBox to the size of the raw image
    view_box = f"0 0 {raw_image_size['width']} {raw_image_size['height']}"
    # SVG canvas size should match the raw image size
    svg_width = raw_image_size['width']
    svg_height = raw_image_size['height']
    group_id = f"eye-ai:{annotation_tag_rid},{annotation_tag_name}"
    rect_stroke_color = "#ff0000"

    svg_content = f'''<svg width="{svg_width}px" height="{svg_height}px" viewBox="{view_box}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
      <g id="{group_id}">
        <rect fill="none" x="{bbox['left']}" y="{bbox['top']}" width="{bbox['width']}" height="{bbox['height']}" stroke="{rect_stroke_color}" stroke-width="2"/>
      </g>
    </svg>
    '''

    svg_file_path = os.path.join(output_directory, f"{annotation_tag_rid}_{rid}.svg")
    os.makedirs(os.path.dirname(svg_file_path), exist_ok=True)
    with open(svg_file_path, "w") as file:
        file.write(svg_content)



def preprocess_and_crop(directory_path, csv_path, output_csv_path, template_path, output_path, model, process_rid, annotation_tag_rid, annotation_tag_name):
    # Template creation
    template = np.ones((50, 50), dtype="uint8") * 0
    template = cv.circle(template, (25, 25), 24, 255, -1)

    if not cv.imwrite(template_path, template):
        print("Error: Template could not be saved.")
        exit()

    def imgResize_primary(img):
        h = img.shape[0]
        w = img.shape[1]
        perc = 500 / w
        w1 = 500
        h1 = int(h * perc)
        img_rs = cv.resize(img, (w1, h1))
        return img_rs

    def imgResize_secondary(img):
        img_rs = cv.resize(img, (600, 600))
        return img_rs

    def getImage(directory_path, img_name):
        full_path = directory_path + img_name
        image = cv.imread(full_path, -1)
        if image is None:
            print("Error: Image could not be read.")
            return None
        return image

    def kmeansclust(img, k):
        img_rsp = img.reshape((-1, 1))
        img_rsp = img_rsp.astype('float32')
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 400, 0.99)
        _, labels, (centers) = cv.kmeans(img_rsp, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        centers = centers.astype('uint8')
        labels = labels.flatten()
        seg_img = centers[labels.flatten()]
        seg_img = seg_img.reshape(img.shape)
        return seg_img

    def imgResizeToFixedSize(img):
        desired_dim = (300, 300) 
        img_rs = cv.resize(img, desired_dim)
        return img_rs

    def predict_optic_disc_center(image, model):
        IMG_SIZE = (300, 300)
        image = cv2.resize(image, IMG_SIZE)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        prediction = model.predict(image)
        return "Proper" if prediction > 0.5 else "Not Proper"

    def crop_to_eye(im):
        mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(max_contour)
        cropped_im = im[y:y+h, x:x+w]
        return cropped_im
    
    def crop_to_eye_with_scaling_to_raw_image(im):
        mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(max_contour)
        cropped_im = im[y:y+h, x:x+w]
        return cropped_im, x, y  # Return the cropped image and the top-left corner coordinates


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    start_time = time.time()

    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist.")
        exit()

    image_files = os.listdir(directory_path)  # List all files in the directory
    csv_data = pd.read_csv(csv_path)

    crop_success = False  # Initialize the flag
    results_data = []  # Initialize the results data list

    for index, row in csv_data.iterrows():
        img_name = row['Filename']
        rid = row['RID'] # 
        image_vocab = row['Image_Tag']

        if img_name not in image_files:
            print(f"Image {img_name} does not exist in directory.")
            continue

        img = getImage(directory_path, img_name)
        if img is None:
            continue
        raw_image_size = {"width": img.shape[1], "height": img.shape[0]}
        resize_functions = [imgResize_primary, imgResize_secondary]  # Put resizing functions in a list
        for crop_size in range(95, 116, 10):
            for resize_function in resize_functions:  # Iterate over resizing functions
                # img = crop_to_eye(img)  # First, crop to eye
                img, raw_crop_x, raw_crop_y = crop_to_eye_with_scaling_to_raw_image(img)  # First, crop to eye
                img_rs = resize_function(img)
                original_scale = img.shape[0] / img_rs.shape[0]  # Calculate the scale of the original image
                for trial, color_channel in enumerate(["grey", "green", "red", "blue"], 1):
                    print(f"Processing image {img_name} : , trial {trial}, color channel {color_channel}, resize function {resize_function.__name__}")
                    if color_channel == "grey":
                        img_k = kmeansclust(cv.cvtColor(img_rs, cv.COLOR_BGR2GRAY), 7)
                    else:
                        img_k = kmeansclust(cv.split(img_rs)[{"red": 2, "green": 1, "blue": 0}[color_channel]], 7)

                    temp = cv.imread(template_path, -1)
                    if temp is None:
                        print("Error: Template could not be read.")
                        continue

                    metd = cv.TM_CCOEFF_NORMED
                    temp_mat = cv.matchTemplate(img_k, temp, metd)

                    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(temp_mat)
                    x = max_loc[0] + 45
                    y = max_loc[1] + 45

                    top = max(0, y - crop_size)
                    bottom = min(img_rs.shape[0], y + crop_size)
                    left = max(0, x - crop_size)
                    right = min(img_rs.shape[1], x + crop_size)

                    # Scale the coordinates back to the original image size
                    top1 = int(top * original_scale)
                    bottom1 = int(bottom * original_scale)
                    left1 = int(left * original_scale)
                    right1 = int(right * original_scale)

                    height1, width1 = bottom1 - top1, right1 - left1
                    if height1 < 600:
                        diff1 = 600 - height1
                        top1 = max(0, top1 - diff1 // 2)
                        bottom1 = min(img.shape[0], bottom1 + (diff1 - diff1 // 2))

                    if width1 < 600:
                        diff1 = 600 - width1
                        left1 = max(0, left1 - diff1 // 2)
                        right1 = min(img.shape[1], right1 + (diff1 - diff1 // 2))

                    cropped_img1 = img[top1:bottom1, left1:right1]
                    resized_img1 = imgResizeToFixedSize(cropped_img1)

                    raw_top1 = int((top * original_scale) + raw_crop_y)
                    raw_bottom1 = int((bottom * original_scale) + raw_crop_y)
                    raw_left1 = int((left * original_scale) + raw_crop_x)
                    raw_right1 = int((right * original_scale) + raw_crop_x)


                    prediction = predict_optic_disc_center(resized_img1, model)  # Replace with your function
                    if prediction == "Proper":

                        img_path1 = f'{output_path}{"Cropped_High_Resolution"}_{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}' 
                        if not cv.imwrite(img_path1, cropped_img1):
                            print(f"Error: Image could not be saved at: {img_path1}")

                        # Save the SVG file
                        
                        bbox = {
                            "left": raw_left1,
                            "top": raw_top1,
                            "width": raw_right1 - raw_left1,
                            "height": raw_bottom1 - raw_top1
                        }

                        save_svg(output_path, annotation_tag_rid, rid, raw_image_size, bbox, annotation_tag_name)
                        


                        print(f"SVG for {rid} saved.")

                        print(f"Image {img_name} ({color_channel}) cropped and saved at {img_path1}.")
                        crop_success = True  # Set the flag

                        # Append the information to the results data list
                        results_data.append({"Image Name": img_name,
                                            "Saved Image Name": f'{"Cropped_High_Resolution"}_{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}',
                                    "Worked Color Channel": color_channel,
                                    "Cumulative Trials": trial,
                                    "Worked Crop Size": crop_size,
                                                     "Bounding Box Top": raw_top1,
                                                     "Bounding Box Bottom": raw_bottom1,
                                                     "Bounding Box Left": raw_left1,
                                                 "Bounding Box Right": raw_right1,
                                    "Worked Image Cropping Function": resize_function.__name__})

                        break
                    else:
                        continue

                if crop_success:
                    break  # Break from the resize_functions loop
            if crop_success:
                break  # Break the loop over crop sizes

        # If all fails, save the raw cropped image
        if not crop_success:
            
            img_rs = crop_to_eye(img)

            # Define the image paths
            raw_img_path = f'{output_path}{"Cropped_High_Resolution"}_{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}' #f'{dir_path}{"Cropped_High_Res"}_{img_name.split(".")[0]}.{img_name.split(".")[1]}'
    
            if not cv.imwrite(raw_img_path, img_rs):
                print(f"Error: Raw image could not be saved at: {raw_img_path}")
            print(f"Raw Image {img_name} saved  at {raw_img_path}.")

                    # Append the information to the DataFrame
            results_data.append({"Image Name": img_name,
                                 "Saved Image Name": f'{"Cropped_High_Resolution"}_{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}',
                                            "Worked Color Channel": "None",
                                            "Worked Image Cropping Function": "Raw Cropped to Eye"},)

        crop_success = False  # Reset the flag for the next image

    
    # delete the template image
    try:
        os.remove(template_path)
    except:
        pass

    # Create DataFrame from results data
    results_df = pd.DataFrame(results_data)
    # output_csv_path = os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_results.csv")
    results_df.to_csv(output_csv_path, index=False)



    print(f"Number of images in CSV: {results_df.shape[0]}")
    print(f"Number of images in directory: {len(image_files)}")
    print(f"Number of images in output directory: {len(os.listdir(output_path))}")
    print(f"Number of cropped images: {len(results_data)}")

    print("--- %s seconds ---" % (time.time() - start_time))


import argparse

# create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--bag_path', type=str, required=True, help='Path to the bag')
parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV')
parser.add_argument('--template_path', type=str, required=False, help='Path to the template')
parser.add_argument('--output_csv_path', type=str, required=True, help='Path to the output CSV')
parser.add_argument('--output_directory', type=str, required=True, help='Path to the output directory')
parser.add_argument('--model_path', type=str, required=False, help='Path to the model')
parser.add_argument('--process_rid', type=str, required=True, help='Process RID')
parser.add_argument('--annotation_tag_rid', type=str, required=True, help='Annotation Tag RID')
parser.add_argument('--annotation_tag_name', type=str, required=True, help='Annotation Tag Name')

# parse the arguments
args = parser.parse_args()

# Define image and output directories
directory_path = args.bag_path #os.path.join(args.bag_path)#, 'data/assets/Image/') # args.bag_path
output_directory = args.output_directory # '/Users/sreenidhi/Downloads/test of. highrez_cropped/'

# Your CSV loading here
csv_path = args.csv_path #f'/Users/sreenidhi/Downloads/Book1.csv'

# Template path
template_path = args.template_path #'/Users/sreenidhi/Downloads/template.jpg'

# Output CSV path
output_csv_path =  args.output_csv_path #f'/Users/sreenidhi/Downloads/Field_2_Cropped_Results_DataFrame.csv'

# model loading here
from Crop_Detector_1_0_0 import load_model
model = load_model(args.model_path)  

# preprocess_and_crop(directory_path, csv_path, output_csv_path, template_path, output_path, model)
preprocess_and_crop(directory_path, csv_path, output_csv_path, template_path, output_directory, model, args.process_rid, args.annotation_tag_rid, args.annotation_tag_name)


'''
python Cleaned_Optic_Disc_Cropping_Algorithm_1_0_3_1.py --bag_path "Input/" --csv_path "Input.csv" --template_path "template.jpg" --output_directory "Output/" --model_path "/Users/sreenidhi/Downloads/USC/HSC Research/EYE AI/Glaucoma or Not Glaucoma/Merged Cropped Porper or Not Dataset TLBR 95 VGG19 Val Accuracy.hdf5" --output_csv_path "Output.csv" --process_rid "PROCESS_RID" --annotation_tag_rid "ANNOTATION_TAG_RID" --annotation_tag_name "ANNOTATION_TAG_NAME"
'''