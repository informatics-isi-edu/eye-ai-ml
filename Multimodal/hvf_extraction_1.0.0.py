from hvf_extraction_script.hvf_data.hvf_object import Hvf_Object
from hvf_extraction_script.utilities.file_utils import File_Utils

hvf_img_path = "/Users/sreenidhi/Downloads/HSC Research/drive-download-20230522T222415Z-001/47389_SOLOMON, JOAN_1935-09-17_2096f43e9a024476/(389819)_2022-05-24_HFA 1 LA (HFA)/IMAGE_0_(2).jpg"
hvf_img = File_Utils.read_image_from_file(hvf_img_path)
hvf_obj = Hvf_Object.get_hvf_object_from_image(hvf_img)

print("KEYLABEL_NAME : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_NAME])
print("KEYLABEL_DOB : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_DOB])

print("KEYLABEL_ID : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_ID])

print("KEYLABEL_FIELD_SIZE : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_FIELD_SIZE]) 

print("KEYLABEL_STRATEGY : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_STRATEGY])


print("KEYLABEL_FALSE_NEG : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_FALSE_NEG])
print("KEYLABEL_FALSE_POS : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_FALSE_POS])
print("KEYLABEL_FIXATION_LOSS : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_FIXATION_LOSS])
print("KEYLABEL_FOVEA : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_FOVEA])
print("KEYLABEL_LATERALITY : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_LATERALITY])
print("KEYLABEL_LAYOUT : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_LAYOUT])
print("KEYLABEL_MD : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_MD])

print("KEYLABEL_PSD : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_PSD])
print("KEYLABEL_PUPIL_DIAMETER : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_PUPIL_DIAMETER])

print("KEYLABEL_RX : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_RX])
print("KEYLABEL_TEST_DATE : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_TEST_DATE])
print("KEYLABEL_TEST_DURATION : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_TEST_DURATION])
print("KEYLABEL_VFI : ", hvf_obj.metadata[Hvf_Object.KEYLABEL_VFI])


import cv2 
import pytesseract
import re

img = cv2.imread(hvf_img_path)

# Adding custom options
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

# Print the results
for key, value in data.items():
    print(f"{key}: {value}")

print("get_display_abs_perc_plot_string : ", hvf_obj.get_display_abs_perc_plot_string())  
print("get_display_abs_val_plot_string : ", hvf_obj.get_display_abs_val_plot_string())
print("get_display_pat_perc_plot_string : ", hvf_obj.get_display_pat_perc_plot_string())
print("get_display_pat_val_plot_string : ", hvf_obj.get_display_pat_val_plot_string())
print("get_display_raw_val_plot_string : ", hvf_obj.get_display_raw_val_plot_string())

