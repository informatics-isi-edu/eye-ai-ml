import os
import cv2 as cv
import numpy as np
import pandas as pd
import time
import tensorflow as tf

IMG_SIZE = (224, 224)

preprocess_input = tf.keras.applications.vgg19.preprocess_input

def preprocess_and_crop(directory_path, csv_path, output_csv_path, template_path, output_path, model):
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
        desired_dim = (224, 224)
        img_rs = cv.resize(img, desired_dim)
        return img_rs

    def predict_optic_disc_center(image, model):
        IMG_SIZE = (224, 224)
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
        rid = row['RID']
        image_vocab = row['Image_Tag']

        if img_name not in image_files:
            print(f"Image {img_name} does not exist in directory.")
            continue

        img = getImage(directory_path, img_name)
        if img is None:
            continue
        resize_functions = [imgResize_primary, imgResize_secondary]  # Put resizing functions in a list

        for resize_function in resize_functions:  # Iterate over resizing functions
            img = crop_to_eye(img)  # First, crop to eye
            img_rs = resize_function(img)
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

                top = max(0, y - 80)
                bottom = min(img_rs.shape[0], y + 80)
                left = max(0, x - 80)
                right = min(img_rs.shape[1], x + 80)

                height, width = bottom - top, right - left
                if height < 224:
                    diff = 224 - height
                    top = max(0, top - diff // 2)
                    bottom = min(img_rs.shape[0], bottom + (diff - diff // 2))

                if width < 224:
                    diff = 224 - width
                    left = max(0, left - diff // 2)
                    right = min(img_rs.shape[1], right + (diff - diff // 2))

                cropped_img = img_rs[top:bottom, left:right]
                resized_img = imgResizeToFixedSize(cropped_img)

                prediction = predict_optic_disc_center(resized_img, model)  # Replace with your function
                if prediction == "Proper":
                    img_path = f'{output_path}{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}'
                    if not cv.imwrite(img_path, resized_img):
                        print(f"Error: Image could not be saved at: {img_path}")
                    print(f"Image {img_name} ({color_channel}) cropped and saved.")
                    crop_success = True  # Set the flag

                    # Append the information to the results data list
                    results_data.append({"Image Name": img_name,
                                         "Saved Image Name": f'{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}',
                                "Worked Color Channel": color_channel,
                                "Cumulative Trials": trial,
                                "Worked Image Cropping Function": resize_function.__name__})

                    break
                else:
                    continue

            if crop_success:
                break  # Break from the resize_functions loop

        # If no successful cropping found for all color channels and resize functions, resize raw image without cropping and try again
        if not crop_success:
            img_rs = imgResize_primary(img)
            for trial, color_channel in enumerate(["grey", "green", "red", "blue"], 1):
                print(f"Processing image {img_name} : , trial {trial}, color channel {color_channel}, without cropping")
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

                top = max(0, y - 80)
                bottom = min(img_rs.shape[0], y + 80)
                left = max(0, x - 80)
                right = min(img_rs.shape[1], x + 80)

                height, width = bottom - top, right - left
                if height < 224:
                    diff = 224 - height
                    top = max(0, top - diff // 2)
                    bottom = min(img_rs.shape[0], bottom + (diff - diff // 2))

                if width < 224:
                    diff = 224 - width
                    left = max(0, left - diff // 2)
                    right = min(img_rs.shape[1], right + (diff - diff // 2))

                cropped_img = img_rs[top:bottom, left:right]
                resized_img = imgResizeToFixedSize(cropped_img)

                prediction = predict_optic_disc_center(resized_img, model)  # Replace with your function
                if prediction == "Proper":
                    img_path = f'{output_path}{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}'
                    if not cv.imwrite(img_path, resized_img):
                        print(f"Error: Image could not be saved at: {img_path}")
                    print(f"Image {img_name} ({color_channel}) resized without cropping and saved.")
                    crop_success = True  # Set the flag

                    # Append the information to the results data list
                    results_data.append({"Image Name": img_name,
                                                    "Worked Color Channel": color_channel,
                                                    "Saved Image Name": f'{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}',
                                                    "Cumulative Trials": trial,
                                                    "Worked Image Cropping Function": "Imgresize2 without cropping"},)

                    break

        # If all fails, save the raw cropped image
        if not crop_success:
            img_rs = crop_to_eye(img)
            raw_img_path = f'{output_path}{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}'
            if not cv.imwrite(raw_img_path, img_rs):
                print(f"Error: Raw image could not be saved at: {raw_img_path}")
            print(f"Raw Image {img_name} saved.")

                    # Append the information to the DataFrame
            results_data.append({"Image Name": img_name,
                                 "Saved Image Name": f'{rid}_{img_name.split(".")[0]}_{image_vocab}.{img_name.split(".")[1]}',
                                            "Worked Color Channel": "None",
                                            # "Cumulative Trials": trial,
                                            "Worked Image Cropping Function": "Raw Cropped to Eye"},)

        crop_success = False  # Reset the flag for the next image


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

# parse the arguments
args = parser.parse_args()

# Define image and output directories
directory_path = os.path.join(args.bag_path, 'data/assets/Image/') # args.bag_path
output_directory = '/content/images/cropped/'

# Your CSV loading here
csv_path = f'/content/{os.path.basename(args.bag_path)}_Field_2.csv'

# Template path
template_path = '/content/template.jpg'

# Output CSV path
output_csv_path =  f'/content/{os.path.basename(args.bag_path)}_Field_2_Cropped_Results_DataFrame.csv'

# model loading here
from Crop_Detector_1_0_0 import load_model
model = load_model('Imgresize2 and Grey data_crop_proper_or_not prof crop data.hdf5')  

# preprocess_and_crop(directory_path, csv_path, output_csv_path, template_path, output_path, model)
preprocess_and_crop(directory_path, csv_path, output_csv_path, template_path, output_directory, model)