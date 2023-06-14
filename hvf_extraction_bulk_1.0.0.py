import os
import csv
from hvf_extraction_script.hvf_data.hvf_object import Hvf_Object
from hvf_extraction_script.utilities.file_utils import File_Utils
import cv2 
import pytesseract
import re

# Path to the directory containing the images
image_dir = "/Users/sreenidhi/Downloads/HSC Research/drive-download-20230522T222415Z-001/HVF"

# CSV output file path
output_file_path = "/Users/sreenidhi/Downloads/HSC Research/drive-download-20230522T222415Z-001/HVF/output.csv"

# Open the CSV file for writing
with open(output_file_path, "w") as csv_file:
    writer = csv.writer(csv_file)
    # Write the headers
    headers = [
        'Image Name', 
        'KEYLABEL_NAME', 
        'KEYLABEL_DOB', 
        'KEYLABEL_ID', 
        'KEYLABEL_FIELD_SIZE', 
        'KEYLABEL_STRATEGY', 
        'KEYLABEL_FALSE_NEG', 
        'KEYLABEL_FALSE_POS', 
        'KEYLABEL_FIXATION_LOSS', 
        'KEYLABEL_FOVEA', 
        'KEYLABEL_LATERALITY', 
        'KEYLABEL_LAYOUT', 
        'KEYLABEL_MD', 
        'KEYLABEL_PSD', 
        'KEYLABEL_PUPIL_DIAMETER', 
        'KEYLABEL_RX', 
        'KEYLABEL_TEST_DATE', 
        'KEYLABEL_TEST_DURATION', 
        'KEYLABEL_VFI', 
        'KEYLABEL_Gender', 
        'KEYLABEL_FIXATION_MONITOR', 
        'KEYLABEL_FIXATION_TARGET', 
        'KEYLABEL_STIMULUS', 
        'KEYLABEL_BACKGROUND', 
        'KEYLABEL_VISUAL_ACUITY', 
        'KEYLABEL_AGE', 
        'KEYLABEL_GHT', 
        'KEYLABEL_PSD',
        'get_display_abs_perc_plot_string',
        'get_display_abs_val_plot_string',
        'get_display_pat_perc_plot_string',
        'get_display_pat_val_plot_string',
        'get_display_raw_val_plot_string',
    ]
    writer.writerow(headers)

    # Iterate through each image in the directory
    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            # Full path to the image
            hvf_img_path = os.path.join(image_dir, image_name)

            # Process the image with hvf_extraction_script
            hvf_img = File_Utils.read_image_from_file(hvf_img_path)
            hvf_obj = Hvf_Object.get_hvf_object_from_image(hvf_img)

            # Process the image with pytesseract
            img = cv2.imread(hvf_img_path)
            custom_config = r'--oem 3 --psm 6'
            ocr_result = pytesseract.image_to_string(img, config=custom_config)

            # Regular expressions for each data point
            regexes = {
                "KEYLABEL_Gender": r"Gender:\s*([A-Za-z]+)",
                "KEYLABEL_FIXATION_MONITOR": r"Fixation Monitor:\s*(.+?)(?=Stimulus)",
                "KEYLABEL_FIXATION_TARGET": r"Fixation Target:\s*(.+?)(?=Background)",
                "KEYLABEL_STIMULUS": r"Stimulus:\s*(.+?)(?=Date)",
                "KEYLABEL_BACKGROUND": r"Background:\s*(.+?)(?=Time)",
                "KEYLABEL_VISUAL_ACUITY": r"Visual Acuity:\s*(.+)",
                "KEYLABEL_AGE": r"Age:\s*(\d+)",
                "KEYLABEL_GHT": r"GHT:\s*(.+)",
                "KEYLABEL_PSD": r"PSD24-2:\s*(.+?)(?=dBP)"
            }

            # Dictionary to store results
            data = {}

            # Loop through each regular expression, and store the first match in the data dictionary
            for key, regex in regexes.items():
                match = re.search(regex, ocr_result)
                if match:
                    data[key] = match.group(1).strip()

            # Prepare the row to be written to the CSV file
            row = [
                image_name, 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_NAME), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_DOB), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_ID), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_FIELD_SIZE), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_STRATEGY),
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_FALSE_NEG), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_FALSE_POS), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_FIXATION_LOSS), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_FOVEA), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_LATERALITY), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_LAYOUT),
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_MD), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_PSD), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_PUPIL_DIAMETER),
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_RX), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_TEST_DATE), 
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_TEST_DURATION),
                hvf_obj.metadata.get(Hvf_Object.KEYLABEL_VFI), 
                data.get('KEYLABEL_Gender'), 
                data.get('KEYLABEL_FIXATION_MONITOR'),
                data.get('KEYLABEL_FIXATION_TARGET'),
                data.get('KEYLABEL_STIMULUS'), 
                data.get('KEYLABEL_BACKGROUND'),
                data.get('KEYLABEL_VISUAL_ACUITY'), 
                data.get('KEYLABEL_AGE'), 
                data.get('KEYLABEL_GHT'),
                data.get('KEYLABEL_PSD'),
                hvf_obj.get_display_abs_perc_plot_string(),
                hvf_obj.get_display_abs_val_plot_string(),
                hvf_obj.get_display_pat_perc_plot_string(),
                hvf_obj.get_display_pat_val_plot_string(),
                hvf_obj.get_display_raw_val_plot_string(),
            ]
            writer.writerow(row)

print(f"Results saved to {output_file_path}")
