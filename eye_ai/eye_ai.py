import xml.etree.ElementTree as ET
from pathlib import Path, PurePath
from typing import List, Callable, Optional
from importlib.metadata import version
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve
from deriva_ml import DerivaML, DerivaMLException, DatasetBag
import tensorflow as tf
import numpy as np

class EyeAIException(DerivaMLException):
    def __init__(self, msg=""):
        super().__init__(msg=msg)


class EyeAI(DerivaML):
    """
    EyeAI is a class that extends DerivaML and provides additional routines for working with eye-ai
    catalogs using deriva-py.

    Attributes:
    - protocol (str): The protocol used to connect to the catalog (e.g., "https").
    - hostname (str): The hostname of the server where the catalog is located.
    - catalog_number (str): The catalog number or name.
    - credential (object): The credential object used for authentication.
    - catalog (ErmrestCatalog): The ErmrestCatalog object representing the catalog.
    - pb (PathBuilder): The PathBuilder object for constructing URL paths.

    Methods:
    - __init__(self, hostname: str = 'www.eye-ai.org', catalog_number: str = 'eye-ai'): Initializes the EyeAI object.
    - create_new_vocab(self, schema_name: str, table_name: str, name: str, description: str, synonyms: List[str] = [],
            exist_ok: bool = False) -> str: Creates a new controlled vocabulary in the catalog.
    - image_tall(self, dataset_rid: str, diagnosis_tag_rid: str): Retrieves tall-format image data based on provided
      diagnosis tag filters.
    - add_process(self, process_name: str, github_url: str = "", process_tag: str = "", description: str = "",
                    github_checksum: str = "", exists_ok: bool = False) -> str: Adds a new process to the Process table.
    - compute_diagnosis(self, df: pd.DataFrame, diag_func: Callable, cdr_func: Callable,
                          image_quality_func: Callable) -> List[dict]: Computes new diagnosis based on
                                                                       provided functions.
    - insert_new_diagnosis(self, entities: List[dict[str, dict]], diagTag_RID: str, process_rid: str): Batch inserts new
      diagnosis entities into the Diagnoisis table.

    Private Methods:
    - _find_latest_observation(df: pd.DataFrame): Finds the latest observations for each subject in the DataFrame.
    - _batch_insert(table: datapath._TableWrapper, entities: Sequence[dict[str, str]]): Batch inserts
       entities into a table.
    """

    def __init__(self, hostname: str = 'www.eye-ai.org', catalog_id: str = 'eye-ai',
                 cache_dir: str = '/data', working_dir: str = None, ml_schema: str = 'deriva-ml'):
        """
        Initializes the EyeAI object.

        Args:
        - hostname (str): The hostname of the server where the catalog is located.
        - catalog_number (str): The catalog number or name.
        """

        super().__init__(hostname = hostname, catalog_id = catalog_id,
                         domain_schema = 'eye-ai', project_name = 'eye-ai',
                         cache_dir = cache_dir, working_dir = working_dir,
                         model_version=version(__name__.split('.')[0]),
                         ml_schema = ml_schema)

    @staticmethod
    def _find_latest_observation(df: pd.DataFrame):
        """
        Filter a DataFrame to retain only the rows representing the latest encounters for each subject.

        Args:
        - df (pd.DataFrame): Input DataFrame containing columns 'Subject_RID' and 'Date_of_Encounter'.

        Returns:
        - pd.DataFrame: DataFrame filtered to keep only the rows corresponding to the latest encounters
          for each subject.
        """
        latest_encounters = {}
        for index, row in df.iterrows():
            subject_rid = row['Subject_RID']
            date_of_encounter = row['date_of_encounter']
            if subject_rid not in latest_encounters or date_of_encounter > latest_encounters[subject_rid]:
                latest_encounters[subject_rid] = date_of_encounter
        for index, row in df.iterrows():
            if row['date_of_encounter'] != latest_encounters[row['Subject_RID']]:
                df.drop(index, inplace=True)
        return df

    def image_tall(self, ds_bag: DatasetBag, diagnosis_tag: str) -> pd.DataFrame:
        """
        Retrieve tall-format image data based on provided dataset and diagnosis tag filters.

        Args:
        - dataset_rid (str): RID of the dataset to filter images.
        - diagnosis_tag (str): Name of the diagnosis tag used for further filtering.

        Returns:
        - pd.DataFrame: DataFrame containing tall-format image data from fist observation of the subject,
          based on the provided filters.
        """

        sys_cols = ['RCT', 'RMT', 'RCB', 'RMB']
        subject = ds_bag.get_table_as_dataframe('Subject').rename(columns={'RID': 'Subject_RID'}).drop(columns=sys_cols)
        observation = ds_bag.get_table_as_dataframe('Observation').rename(columns={'RID': 'Observation_RID'}).drop(columns=sys_cols)
        image = ds_bag.get_table_as_dataframe('Image').rename(columns={'RID': 'Image_RID'}).drop(columns=sys_cols)
        diagnosis = ds_bag.get_table_as_dataframe('Image_Diagnosis').rename(columns={'RID': 'Diagnosis_RID'}).drop(columns=['RCT', 'RMT', 'RMB'])
        
        merge_obs = pd.merge(subject, observation, left_on='Subject_RID', right_on='Subject', how='left')
        merge_image = pd.merge(merge_obs, image, left_on='Observation_RID', right_on='Observation', how='left')
        merge_diag = pd.merge(merge_image, diagnosis, left_on='Image_RID', right_on='Image', how='left')
        image_frame = merge_diag[merge_diag['Image_Angle'] == '2']

        image_frame = image_frame[image_frame['Diagnosis_Tag'] == diagnosis_tag]
        # Select only the first observation which included in the grading app.
    
     
        image_frame = self._find_latest_observation(image_frame)
   
        grading_tags = ["GlaucomaSuspect", "AI_glaucomasuspect_test",
                        "GlaucomaSuspect-Training", "GlaucomaSuspect-Validation"]
        if diagnosis_tag in grading_tags:
            image_frame = pd.merge(image_frame, pd.DataFrame(self.user_list()), how="left", left_on='RCB', right_on='ID')
        else:
            image_frame = image_frame.assign(Full_Name=diagnosis_tag)

        return image_frame[
            ['Subject_RID', 'Image_RID', 'Diagnosis_RID', 'Full_Name', 'Image_Side',
             'Diagnosis_Image', 'Cup_Disk_Ratio', 'Image_Quality']]

    @staticmethod
    def reshape_table(frames: List[pd.DataFrame], compare_value: str):
        """
        Reshape a list of dataframes to long and wide format containing the pre-specified compare value.

        Args:
        - frames (List): A list of dataframes with tall-format image data from fist observation of the subject
        - compare_value (str): Column name of the compared value, choose from ["Diagnosis", "Image_Quality", "Cup_Disk_Ratio"]

        Returns:
        - pd.DataFrame: long and wide formatted dataframe with compare values from all graders and initial diagnosis.
        """
        long = pd.concat(frames).reset_index()
        # change data type for control vocab table
        cols = ['Image_Quality', 'Image_Side', 'Full_Name', 'Diagnosis_Image']
        for c in cols:
            long[c] = long[c].astype('category')
        wide = pd.pivot(long, index=['Image_RID', 'Image_Side', 'Subject_RID'], columns='Full_Name',
                        values=compare_value)  # Reshape from long to wide
        return long, wide

    @staticmethod
    def compute_diagnosis(df: pd.DataFrame,
                          diag_func: Callable,
                          cdr_func: Callable,
                          image_quality_func: Callable) -> List[dict]:
        """
        Compute a new diagnosis based on provided functions.

        Args:
        - df (DataFrame): Input DataFrame containing relevant columns.
        - diag_func (Callable): Function to compute Diagnosis.
        - cdr_func (Callable): Function to compute Cup_Disk Ratio.
        - image_quality_func (Callable): Function to compute Image Quality.

        Returns:
        - List[Dict[str, Union[str, float]]]: List of dictionaries representing the generated Diagnosis.
          The Cup_Disk_Ratio is always round to 4 decimal places.
        """
    
        df["Cup_Disk_Ratio"].replace("", np.nan, inplace=True) 
        df["Cup_Disk_Ratio"] = pd.to_numeric(df["Cup_Disk_Ratio"], errors="coerce")
        result = df.groupby("Image_RID").agg({"Cup_Disk_Ratio": cdr_func,
                                              "Diagnosis_Image": diag_func,
                                              "Image_Quality": image_quality_func})
        result = result.round({'Cup_Disk_Ratio': 4})
        result = result.fillna('NaN')
        result.reset_index('Image_RID', inplace=True)

        return result.to_dict(orient='records')


    @staticmethod
    def filter_angle_2(ds_bag: DatasetBag) -> pd.DataFrame:
        """
        Filters images for just Field_2 and saves the filtered data to a CSV file.

        Parameters:
        - ds_bag (str): DatasetBag of EyeAI dataset.

        Returns:
        - str: Path to the generated CSV file containing filtered images.
        """
        full_set = ds_bag.get_table_as_dataframe('Image')
        dataset_field_2 = full_set[full_set['Image_Angle'] == "2"]
        return dataset_field_2

    @staticmethod
    def get_bounding_box(svg_path: Path) -> tuple:
        """
        Retrieves the bounding box coordinates from an SVG file.

        Parameters:
        - svg_path (str): Path to the SVG file.

        Returns:
        - tuple: A tuple containing the bounding box coordinates (x_min, y_min, x_max, y_max).
        """
        tree = ET.parse(svg_path)
        root = tree.getroot()
        rect = root.find(".//{http://www.w3.org/2000/svg}rect")
        x_min = int(rect.attrib['x'])
        y_min = int(rect.attrib['y'])
        width = int(rect.attrib['width'])
        height = int(rect.attrib['height'])
        bbox = (x_min, y_min, x_min + width, y_min + height)
        return bbox

    def create_cropped_images(self,  ds_bag: DatasetBag, output_dir: Path, crop_to_eye: bool,
                              exclude_list: Optional[list] = None, include_only_list: Optional[list] = None) -> tuple:
        """
        Retrieves images and saves them to the specified directory and separated into two folders by class. Optionally choose to crop the images or not.

        Parameters:
        - ds_bag (DatasetBag): DatasetBag object of the dataset.
        - output_dir(Path): Directory location to save the images.
        - crop_to_eye (bool): Flag indicating whether to crop images to the eye.
        - exclude_list(list): A list of RID to be excluded.
        - include_only_list(list): A list of RID to be included only. Only taking the RID in this list from ds_bag. RIDs in exclude list would still be excluded

        Returns:
        - tuple: A tuple containing the path to the directory containing images and the path to the output CSV file.
        """

        if not exclude_list:
            exclude_list = []

        if not include_only_list:
            include_only_list = []

        out_path_no_glaucoma = output_dir / 'No_Glaucoma'
        out_path_no_glaucoma.mkdir(parents=True, exist_ok=True)
        out_path_glaucoma = output_dir / 'Suspected_Glaucoma'
        out_path_glaucoma.mkdir(parents=True, exist_ok=True)
        
        image_annot_df = ds_bag.get_table_as_dataframe('Annotation')
        image_df = ds_bag.get_table_as_dataframe('Image')
        diagnosis = ds_bag.get_table_as_dataframe('Image_Diagnosis')
        image_bounding_box_df = ds_bag.get_table_as_dataframe('Fundus_Bounding_Box')

        for index, row in image_annot_df.iterrows():
            image_rid = row['Image']
            if include_only_list and image_rid not in include_only_list:
                continue
                    
            if image_rid in exclude_list:
                continue
                
            image_file_path = image_df[image_df['RID'] == image_rid]['Filename'].values[0]
            image_file_name = Path(image_file_path).name
            
            if ds_bag.dataset_rid not in image_file_path:
                print("Error: Image does not belongs to the dataset")
                continue
                
            image = Image.open(str(image_file_path))
            diag = diagnosis[(diagnosis['Diagnosis_Tag'] == 'Initial Diagnosis')
                                     & (diagnosis['Image'] == image_rid)]['Diagnosis_Image'].iloc[0]
            
            out_path_dir = str(out_path_no_glaucoma) if diag == 'No Glaucoma' else str(out_path_glaucoma)
            
            annotation_bounding_box =  pd.merge(image_annot_df[['Image', 'Fundus_Bounding_Box']], 
                                                image_bounding_box_df, 
                                                left_on='Fundus_Bounding_Box', 
                                                right_on='RID')
         
            svg_path = annotation_bounding_box.loc[annotation_bounding_box['Image'] == image_rid, 'Filename'].values[0]

                
            svg_path = Path(svg_path)
            if not svg_path.exists():
                continue
                
            if crop_to_eye:
                bbox = self.get_bounding_box(svg_path)
                cropped_image = image.crop(bbox)
                cropped_image.save(f'{out_path_dir}/Cropped_{image_rid}.JPG')
                image_annot_df.loc[index, 'Cropped Filename'] = 'Cropped_' + image_file_name
            else:
                image.save(f'{str(out_path_dir)}/{image_rid}.JPG')
                image_annot_df.loc[index, 'Filename'] =  image_file_name
                
        image_csv = 'Cropped_Image.csv' if crop_to_eye else 'Image.csv'
        output_csv = output_dir / image_csv
        image_annot_df.to_csv(output_csv)
        
        return output_dir, output_csv

    def create_retfound_image_directory(self,ds_bag_train_dict: dict, 
                                             ds_bag_val_dict: dict, 
                                             ds_bag_test_dict: dict, 
                                             output_dir: Path, 
                                             crop_to_eye: bool = False) -> tuple:
        """
        Wrapper for create_cropped_images to create correct RETFound directory format.

        Parameters:
        - ds_bag_train_dict (dict): A dictionary contains training DatasetBag.
        - ds_bag_val_dict (dict): A dictionary contains validating DatasetBag.
        - ds_bag_test_dict (dict): A dictionary contains testing DatasetBag.
        - output_dir(Path): Directory location to save the images.
        - crop_to_eye (bool): Flag indicating whether to crop images to the eye.
  
        Returns:
        - tuple: A tuple containing the path to the directory containing images and the path to the output CSV file.
        """
     
        if ds_bag_train_dict is None or ds_bag_val_dict is None or ds_bag_test_dict is None:
            print("Error: RETFound required all three train, val, test to be presented")
            return
        
        ds_bag_train = ds_bag_train_dict["ds_bag"]
        ds_bag_val = ds_bag_val_dict["ds_bag"]
        ds_bag_test = ds_bag_test_dict["ds_bag"]

        concat_name = f"{ds_bag_train.dataset_rid}_{ds_bag_val.dataset_rid}_{ds_bag_test.dataset_rid}"
        dir_name = f"{concat_name}_RETFound_Cropped" if crop_to_eye else f"{concat_name}_RETFound"

        output_dir = output_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        train_dir , train_csv = self.create_cropped_images(ds_bag = ds_bag_train, 
                                            output_dir =  output_dir / "train", 
                                            crop_to_eye =  crop_to_eye,
                                            exclude_list = ds_bag_train_dict.get("exclude_list", []), 
                                            include_only_list= ds_bag_train_dict.get("include_list", [])) 
        
        val_dir , val_csv = self.create_cropped_images(ds_bag = ds_bag_val, 
                                            output_dir =  output_dir / "val", 
                                            crop_to_eye =  crop_to_eye,
                                            exclude_list = ds_bag_val_dict.get("exclude_list", []), 
                                            include_only_list= ds_bag_val_dict.get("include_list", [])) 
        
        test_dir , test_csv = self.create_cropped_images(ds_bag = ds_bag_test, 
                                            output_dir =  output_dir / "test", 
                                            crop_to_eye =  crop_to_eye,
                                            exclude_list = ds_bag_test_dict.get("exclude_list", []), 
                                            include_only_list= ds_bag_test_dict.get("include_list", [])) 


        return output_dir, train_dir, train_csv, val_dir, val_csv, test_dir, test_csv


    def plot_roc(self, configuration_record, data: pd.DataFrame) -> Path:
        """
        Plot Receiver Operating Characteristic (ROC) curve based on prediction results. Save the plot values into a csv file.

        Parameters:
        - data (pd.DataFrame): DataFrame containing prediction results with columns 'True Condition_Condition_Condition_Condition_Label' and
        'Probability Score'.
        Returns:
            Path: Path to the saved csv file of ROC plot values .

        """
        output_path = configuration_record.execution_asset_path("ROC")
        pred_result = pd.read_csv(data)
        y_true = pred_result['True Label']
        scores = pred_result['Probability Score']
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
        roc_csv_path = output_path / Path("roc_plot.csv")
        roc_df.to_csv(roc_csv_path, index=False)
        # show plot in notebook
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

        return roc_csv_path

    def compute_condition_label(self, icd10_asso: pd.DataFrame) -> pd.DataFrame:
        icd_mapping = {
            'H40.00*': 'GS',
            'H40.01*': 'GS',
            'H40.02*': 'GS',
            'H40.03*': 'GS',
            'H40.04*': 'GS',
            'H40.05*': 'GS',
            'H40.06*': 'GS',
            'H40.10*': 'POAG',
            'H40.11*': 'POAG',
            'H40.12*': 'POAG',
            'H40.13*': 'POAG',
            'H40.14*': 'POAG',
            'H40.15*': 'POAG',
            'H40.2*': 'PACG'
        }

        def map_icd_to_category(icd_code):
            for key, value in icd_mapping.items():
                if icd_code.startswith(key[:-1]):
                    return value
            return 'Other'

        # Apply the mapping
        icd10_asso['Condition_Label'] = icd10_asso['ICD10_Eye'].apply(map_icd_to_category)
        # Select severity
        priority = {'PACG': 1, 'POAG': 2, 'GS': 3, 'Other': 4}
        icd10_asso['Priority'] = icd10_asso['Condition_Label'].map(priority)
        icd10_asso = icd10_asso.sort_values(by=['Clinical_Records', 'Priority'])
        combined_prior = icd10_asso.drop_duplicates(subset=['Clinical_Records'], keep='first')
        combined_prior = combined_prior.drop(columns=['RID', 'ICD10_Eye', 'Priority'])
        return combined_prior

    def insert_condition_label(self, condition_label: pd.DataFrame):
        condition_label.rename(columns={'Clinical_Records': 'RID'}, inplace=True)
        entities = condition_label.to_dict(orient='records')
        self.domain_path.Clinical_Records.insert(entities)

    @staticmethod
    def _select_24_2(hvf: pd.DataFrame) -> pd.DataFrame:
        hvf_clean = hvf.dropna(subset=['RID_HVF_OCR'])
        priority = {'24-2': 1, '10-2': 2, '30-2': 3}
        hvf_clean.loc[:, 'priority'] = hvf_clean['Field_Size'].map(priority)
        hvf_sorted = hvf_clean.sort_values(by=['RID_Observation', 'priority'])
        result = hvf_sorted.groupby(['RID_Observation', 'Image_Side']).first().reset_index()
        result = result.drop(columns=['priority'])
        return result

    @staticmethod
    def closest_to_fundus(report, fundus):
        report['date_of_encounter'] = pd.to_datetime(report['date_of_encounter']).dt.tz_localize(None)
        fundus['date_of_encounter'] = pd.to_datetime(fundus['date_of_encounter']).dt.tz_localize(None)
        report_match = pd.DataFrame()

        def find_closest_date(target_date, dates):
            return min(dates, key=lambda d: abs(d - target_date))

        for idx, row in fundus.iterrows():
            rid = row['RID_Subject']
            target_date = row['date_of_encounter']

            for side in ['Left', 'Right']:
                filtered_data = report[(report['RID_Subject'] == rid) & (report['Image_Side'] == side)]
                if not filtered_data.empty:
                    # Find the closest date entry
                    if sum(filtered_data['date_of_encounter'].isna()) > 0:
                        report_match = pd.concat([report_match, filtered_data.iloc[[0]]])
                    else:
                        closest_date = find_closest_date(target_date, filtered_data['date_of_encounter'])
                        closest_entries = filtered_data[filtered_data['date_of_encounter'] == closest_date]
                        report_match = pd.concat([report_match, closest_entries])
        return report_match

    def extract_modality(self, ds_bag: DatasetBag) -> dict[str, pd.DataFrame]:
        sys_cols = ['RCT', 'RMT', 'RCB', 'RMB']
        subject = ds_bag.get_table_as_dataframe('Subject').drop(columns=sys_cols)
        observation = ds_bag.get_table_as_dataframe('Observation')[['RID', 'Observation_ID', 'Subject', 'date_of_encounter']]
        image = ds_bag.get_table_as_dataframe('Image').drop(columns=sys_cols)
        observation_clinic_asso = ds_bag.get_table_as_dataframe('Clinical_Records_Observation').drop(columns=sys_cols)
        clinic = ds_bag.get_table_as_dataframe('Clinical_Records').drop(columns=sys_cols)
        report = ds_bag.get_table_as_dataframe('Report').drop(columns=sys_cols)
        rnfl_ocr = ds_bag.get_table_as_dataframe('OCR_RNFL').drop(columns=sys_cols)
        hvf_ocr = ds_bag.get_table_as_dataframe('OCR_HVF').drop(columns=sys_cols)

        subject_observation = pd.merge(subject, observation, left_on='RID', right_on='Subject', how='left',
                                       suffixes=('_Subject', '_Observation')).drop(columns=['Subject'])

        # Report_HVF
        subject_observation_report = pd.merge(subject_observation, report,
                                              left_on='RID_Observation',
                                              right_on='Observation',
                                              suffixes=("subject_observation_for_HVF", "Report")).drop(
            columns=['Observation']).rename(columns={'RID': 'RID_Report'})
        hvf = pd.merge(subject_observation_report, hvf_ocr,
                       left_on='RID_Report',
                       right_on='Report',
                       suffixes=("_subject_observation_for_HVF_report", "_HVF_OCR"),
                       how='left').rename(columns={'RID': 'RID_HVF_OCR'}).drop(columns=['URL', 'Description',
                                                                                        'Length', 'MD5', 'Report'])

        hvf = self._select_24_2(hvf)

        # Report_RNFL
        rnfl = pd.merge(subject_observation_report, rnfl_ocr,
                        left_on='RID_Report',
                        right_on='Report',
                        suffixes=("_subject_observation_for_RNFL_report", "_RNFL_OCR"),
                        how='left').rename(columns={'RID': 'RID_RNFL_OCR'}).drop(columns=['URL', 'Description',
                                                                                          'Length', 'MD5', 'Report'])

        def highest_signal_strength(rnfl):
            rnfl_clean = rnfl.dropna(subset=['RID_RNFL_OCR', 'Signal_Strength'])
            idx = rnfl_clean.groupby(['RID_Observation', 'Image_Side'])['Signal_Strength'].idxmax()
            result = rnfl_clean.loc[idx]
            return result

        rnfl = highest_signal_strength(rnfl)
        # Image
        image = pd.merge(subject_observation, image,
                         left_on='RID_Observation',
                         right_on='Observation',
                         suffixes=("_subject_observation_for_image",
                                   "_Image")).rename(columns={'RID': 'RID_Image'}).drop(columns=['Observation'])

        # Select the observation according fundus date of encounter
        fundus = image[['RID_Subject', 'Subject_ID', 'Subject_Gender', 'Subject_Ethnicity', 'RID_Observation', 'Observation_ID',
                        'date_of_encounter']].drop_duplicates()

        hvf_match = self.closest_to_fundus(hvf, fundus)
        rnfl_match = self.closest_to_fundus(rnfl, fundus)
        
        # select clinic records by the date of encounter (on the fundus date of encounter)
        subject_obs_clinic = (pd.merge(fundus,
                                       observation_clinic_asso,
                                       left_on='RID_Observation',
                                       right_on='Observation',
                                       how='left').drop(columns=['RID', 'Observation']))
        subject_obs_clinic_data = pd.merge(subject_obs_clinic,
                                           clinic,
                                           left_on='Clinical_Records',
                                           right_on='RID',
                                           suffixes=("", "_Clinic"),
                                           how='left').drop(
            columns=['Clinical_Records']).rename(columns={'RID': 'RID_Clinic',
                                                          'date_of_encounter': 'date_of_encounter_Observation',
                                                          'Date_of_Encounter': 'date_of_encounter_Clinic'})
        clinic_match = subject_obs_clinic_data[
            ['RID_Subject', 'Subject_ID', 'Subject_Gender', 'Subject_Ethnicity', 'RID_Observation',
             'Observation_ID', 'date_of_encounter_Observation', 'RID_Clinic',
             'date_of_encounter_Clinic', 'LogMAR_VA', 'Visual_Acuity_Numerator', 'IOP',
             'Refractive_Error', 'CCT', 'CDR', 'Gonioscopy', 'Condition_Display', 'Provider',
             'Clinical_ID', 'Powerform_Laterality', 'Condition_Label']]

        rnfl_match.rename(columns={'date_of_encounter': 'date_of_encounter_RNFL'}, inplace=True)
        hvf_match.rename(columns={'date_of_encounter': 'date_of_encounter_HVF'}, inplace=True)
        fundus.rename(columns={'date_of_encounter': 'date_of_encounter_Fundus'}, inplace=True)
        return {"Clinic": clinic_match, "HVF": hvf_match, "RNFL": rnfl_match, "Fundus": fundus}

    def multimodal_wide(self, ds_bag: DatasetBag):
        # Todo add fundus image paths
        modality_df = self.extract_modality(ds_bag)
        clinic = modality_df['Clinic'].rename(columns={'Powerform_Laterality': 'Image_Side'})
        rnfl = modality_df['RNFL']
        fundus = modality_df['Fundus']
        hvf = modality_df['HVF']
        
        rid_subjects = pd.concat([
            clinic['RID_Subject'],
            rnfl['RID_Subject'],
            fundus['RID_Subject'],
            hvf['RID_Subject']
        ]).drop_duplicates().reset_index(drop=True)
        sides = pd.DataFrame({'Image_Side': ['Right', 'Left']})
        expanded_subjects = rid_subjects.to_frame().merge(sides, how='cross')
        
        clinic.drop(columns=['RID_Observation', 'Observation_ID', 'date_of_encounter_Observation'], inplace=True)
        rnfl.drop(columns=['RID_Observation', 'Observation_ID'], inplace=True)
        hvf.drop(columns=['RID_Observation', 'Observation_ID'], inplace=True)
        fundus.drop(columns=['RID_Observation', 'Observation_ID'], inplace=True)
        multimodal_wide = pd.merge(expanded_subjects, fundus, how='left', on=['RID_Subject'])
        multimodal_wide = pd.merge(multimodal_wide, clinic, how='left',
                                   on=['RID_Subject', 'Image_Side', 'Subject_ID', 'Subject_Gender', 'Subject_Ethnicity'])
        multimodal_wide = pd.merge(multimodal_wide, hvf, how='left',
                                   on=['RID_Subject', 'Subject_ID', 'Subject_Gender', 'Subject_Ethnicity', 'Image_Side'])
        multimodal_wide = pd.merge(multimodal_wide, rnfl, how='left',
                                   on=['RID_Subject', 'Subject_ID', 'Subject_Gender', 'Subject_Ethnicity', 'Image_Side'],
                                   suffixes=('_HVF', '_RNFL'))
        return multimodal_wide

    def get_multimodal_tf_dataset(self, ds_bag: DatasetBag):
        modality_df = self.extract_modality(ds_bag)
