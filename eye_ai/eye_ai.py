import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePath
from typing import List, Callable, Optional

from itertools import islice
from importlib.metadata import version
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve
from sklearn import preprocessing
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from deriva_ml import DerivaML, DerivaMLException, FileUploadState, UploadState, DatasetBag

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
        self.ml_schema_instance = self.catalog.getPathBuilder().schemas[self.ml_schema]
        self.domain_schema_instance = self.catalog.getPathBuilder().schemas[self.domain_schema]

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

        # Show grader name
        grading_tags = ["GlaucomaSuspect", "AI_glaucomasuspect_test",
                        "GlaucomaSuspect-Training", "GlaucomaSuspect-Validation"]
        if diagnosis_tag in grading_tags:
            image_frame = pd.merge(image_frame, pd.DataFrame(self.user_list()), how="left", left_on='RCB', right_on='ID')
        else:
            image_frame = image_frame.assign(Full_Name=diagnosis_tag)

        return image_frame[
            ['Subject_RID', 'Image_RID', 'Diagnosis_RID', 'Full_Name', 'Image_Side',
             'Diagnosis_Image', 'Cup_Disk_Ratio', 'Image_Quality']]

    def reshape_table(self, frames: List[pd.DataFrame], compare_value: str):
        """
        Reshape a list of dataframes to long and wide format containing the pre-specified compare value.

        Args:
        - frames (List): A list of dataframes with tall-format image data from fist observation of the subject
        - compare_value (str): Column name of the compared value, choose from ["Diagnosis", "Image_Quality", "Cup/Disk_Ratio"]

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

    def compute_diagnosis(self,
                          df: pd.DataFrame,
                          diag_func: Callable,
                          cdr_func: Callable,
                          image_quality_func: Callable) -> List[dict]:
        """
        Compute a new diagnosis based on provided functions.

        Args:
        - df (DataFrame): Input DataFrame containing relevant columns.
        - diag_func (Callable): Function to compute Diagnosis.
        - cdr_func (Callable): Function to compute Cup/Disk Ratio.
        - image_quality_func (Callable): Function to compute Image Quality.

        Returns:
        - List[Dict[str, Union[str, float]]]: List of dictionaries representing the generated Diagnosis.
          The Cup/Disk_Ratio is always round to 4 decimal places.
        """
        df["Cup/Disk_Ratio"] = pd.to_numeric(df["Cup_Disk_Ratio"], errors="coerce")
        result = df.groupby("Image_RID").agg({"Cup_Disk_Ratio": cdr_func,
                                              "Diagnosis_Image": diag_func,
                                              "Image_Quality": image_quality_func})
        result = result.round({'Cup_Disk_Ratio': 4})
        result = result.fillna('NaN')
        result.reset_index('Image_RID', inplace=True)

        return result.to_dict(orient='records')


    def filter_angle_2(self, ds_bag: DatasetBag) -> pd.DataFrame:
        """
        Filters images for just Field_2 and saves the filtered data to a CSV file.

        Parameters:
        - bag_path (str): Path to the bag directory.

        Returns:
        - str: Path to the generated CSV file containing filtered images.
        """
        full_set = ds_bag.get_table_as_dataframe('Image')
        dataset_field_2 = full_set[full_set['Image_Angle'] == "2"]
        return dataset_field_2

    def get_bounding_box(self, svg_path: Path) -> tuple:
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

    def create_cropped_images(self, bag_path: Path, ds_bag: DatasetBag, output_dir: Path, crop_to_eye: bool,
                              exclude_list: Optional[list] = None) -> tuple:
        """
        Retrieves cropped images and saves them to the specified directory and seperated in two folders by class.

        Parameters:
        - bag_path (str): Path to the bag directory.
        - crop_to_eye (bool): Flag indicating whether to crop images to the eye.

        Returns:
        - tuple: A tuple containing the path to the directory containing cropped images and the path to the output CSV file.
        """

        if not exclude_list:
            exclude_list = []
        cropped_path = output_dir / "Image_cropped"
        cropped_path_no_glaucoma = cropped_path / "No_Glaucoma"
        cropped_path_no_glaucoma.mkdir(parents=True, exist_ok=True)
        cropped_path_glaucoma = cropped_path / "Suspected_Glaucoma"
        cropped_path_glaucoma.mkdir(parents=True, exist_ok=True)
        svg_root_path = bag_path / 'data/assets/Fundus_Bounding_Box'
        image_annot_df = ds_bag.get_table_as_dataframe('Annotation')
        image_df = ds_bag.get_table_as_dataframe('Image')
        diagnosis = ds_bag.get_table_as_dataframe('Image_Diagnosis')

        for index, row in image_annot_df.iterrows():
            if row['Annotation_Function'] != 'Raw_Cropped_to_Eye' or crop_to_eye:
                image_rid = row['Image']
                if image_rid not in exclude_list:
                    svg_path = svg_root_path / f'Cropped_{image_rid}.svg'
                    bbox = self.get_bounding_box(svg_path)
                    image_file_name = image_df[image_df['RID'] == image_rid]['Filename'].values[0]
                    image_file_path = bag_path / image_file_name
                    image = Image.open(str(image_file_path))
                    cropped_image = image.crop(bbox)
                    diag = diagnosis[(diagnosis['Diagnosis_Tag'] == 'Initial Diagnosis')
                                     & (diagnosis['Image'] == image_rid)]['Diagnosis_Image'].iloc[0]
                    if diag == 'No Glaucoma':
                        cropped_image.save(f'{str(cropped_path_no_glaucoma)}/Cropped_{image_rid}.JPG')
                    else:
                        cropped_image.save(f'{str(cropped_path_glaucoma)}/Cropped_{image_rid}.JPG')
                    image_annot_df.loc[index, 'Cropped Filename'] = 'Cropped_' + image_file_name
        output_csv = PurePath(self.working_dir, 'Cropped_Image.csv')
        image_annot_df.to_csv(output_csv)
        return cropped_path, output_csv

    def plot_roc(self, configuration_record, data: pd.DataFrame) -> Path:
        """
        Plot Receiver Operating Characteristic (ROC) curve based on prediction results. Save the plot values into a csv file.

        Parameters:
        - data (pd.DataFrame): DataFrame containing prediction results with columns 'True Label' and
        'Probability Score'.
        Returns:
            Path: Path to the saved csv file of ROC plot values .

        """
        output_path = configuration_record.execution_assets_path("ROC")
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

    def _batch_update(self, table, entities):
        """
        Batch update entities in a table.

        Args:
        - table (datapath._TableWrapper): Table wrapper object.
        - entities (Sequence[dict]): Sequence of entity dictionaries to update, must include RID.
        - update_cols (List[datapath._ColumnWrapper]): List of columns to update.

        """

        it = iter(entities)
        while chunk := list(islice(it, 2000)):
            columns = [table.columns[c] for e in chunk for c in e.keys() if c != "RID"]
            table.update(chunk, [table.RID], columns)

    def insert_condition_label(self, condition_label: pd.DataFrame):
        condition_label.rename(columns={'Clinical_Records': 'RID'}, inplace=True)
        entities = condition_label.to_dict(orient='records')
        self._batch_update(self.domain_schema_instance.Clinical_Records, entities)

    def extract_modality(self, ds_bag: DatasetBag):
        sys_cols = ['RCT', 'RMT', 'RCB', 'RMB']
        subject = ds_bag.get_table_as_dataframe('Subject').drop(columns=sys_cols)
        observation = ds_bag.get_table_as_dataframe('Observation')[['RID', 'Observation_ID', 'Subject', 'date_of_encounter']]
        image = ds_bag.get_table_as_dataframe('Image').drop(columns=sys_cols)
        observation_clinic_asso = ds_bag.get_table_as_dataframe('Clinical_Records_Observation').drop(columns=sys_cols)
        clinic = ds_bag.get_table_as_dataframe('Clinical_Records').drop(columns=sys_cols)
        report = ds_bag.get_table_as_dataframe('Report').drop(columns=sys_cols)
        RNFL_OCR = ds_bag.get_table_as_dataframe('OCR_RNFL').drop(columns=sys_cols)
        HVF_OCR = ds_bag.get_table_as_dataframe('OCR_HVF').drop(columns=sys_cols)

        subject_observation = pd.merge(subject, observation, left_on='RID', right_on='Subject', how='left',
                                       suffixes=('_Subject', '_Observation')).drop(columns=['Subject'])

        # Report_HVF
        subject_observation_report = pd.merge(subject_observation, report,
                                              left_on='RID_Observation',
                                              right_on='Observation',
                                              suffixes=("subject_observation_for_HVF", "Report")).drop(
            columns=['Observation']).rename(columns={'RID': 'RID_Report'})
        HVF = pd.merge(subject_observation_report, HVF_OCR,
                       left_on='RID_Report',
                       right_on='Report',
                       suffixes=("_subject_observation_for_HVF_report", "_HVF_OCR"),
                       how='left').rename(columns={'RID': 'RID_HVF_OCR'}).drop(columns=['URL', 'Description',
                                                                                        'Length', 'MD5', 'Report'])

        def select_24_2(HVF):
            HVF_clean = HVF.dropna(subset=['RID_HVF_OCR'])
            priority = {'24-2': 1, '10-2': 2, '30-2': 3}
            HVF_clean['priority'] = HVF_clean['Field_Size'].map(priority)
            HVF_sorted = HVF_clean.sort_values(by=['RID_Observation', 'priority'])
            result = HVF_sorted.groupby(['RID_Observation', 'Image_Side']).first().reset_index()
            result = result.drop(columns=['priority'])
            return result

        HVF = select_24_2(HVF)

        # Report_RNFL
        RNFL = pd.merge(subject_observation_report, RNFL_OCR,
                        left_on='RID_Report',
                        right_on='Report',
                        suffixes=("_subject_observation_for_RNFL_report", "_RNFL_OCR"),
                        how='left').rename(columns={'RID': 'RID_RNFL_OCR'}).drop(columns=['URL', 'Description',
                                                                                          'Length', 'MD5', 'Report'])

        def highest_signal_strength(RNFL):
            RNFL_clean = RNFL.dropna(subset=['RID_RNFL_OCR', 'Signal_Strength'])
            idx = RNFL_clean.groupby(['RID_Observation', 'Image_Side'])['Signal_Strength'].idxmax()
            result = RNFL_clean.loc[idx]
            return result

        RNFL = highest_signal_strength(RNFL)
        # Image
        image = pd.merge(subject_observation, image,
                         left_on='RID_Observation',
                         right_on='Observation',
                         suffixes=("_subject_observation_for_image",
                                   "_Image")).rename(columns={'RID': 'RID_Image'}).drop(columns=['Observation'])

        # Select the observation according fundus date of encounter
        fundus = image[['RID_Subject', 'Subject_ID', 'Subject_Gender', 'Subject_Ethnicity', 'RID_Observation', 'Observation_ID',
                        'date_of_encounter']].drop_duplicates()

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

        HVF_match = closest_to_fundus(HVF, fundus)
        RNFL_match = closest_to_fundus(RNFL, fundus)
        
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

        RNFL_match.rename(columns={'date_of_encounter': 'date_of_encounter_RNFL'}, inplace=True)
        HVF_match.rename(columns={'date_of_encounter': 'date_of_encounter_HVF'}, inplace=True)
        fundus.rename(columns={'date_of_encounter': 'date_of_encounter_Fundus'}, inplace=True)

        # Save df
        clinic_path = PurePath(self.working_dir, 'clinic.csv')
        clinic_match.to_csv(clinic_path, index=False)
        HVF_path = PurePath(self.working_dir, 'HVF.csv')
        HVF_match.to_csv(HVF_path, index=False)
        RNFL_path = PurePath(self.working_dir, 'RNFL.csv')
        RNFL_match.to_csv(RNFL_path, index=False)
        fundus_path = PurePath(self.working_dir, 'fundus.csv')
        fundus.to_csv(fundus_path, index=False)
        return {"Clinic": clinic_path, "HVF": HVF_path, "RNFL": RNFL_path, "Fundus": fundus_path}

    def multimodal_wide(self, ds_bag: DatasetBag):
        modality_df = self.extract_modality(ds_bag)
        Clinic = pd.read_csv(modality_df['Clinic']).rename(columns={'Powerform_Laterality': 'Image_Side'})
        RNFL = pd.read_csv(modality_df['RNFL'])
        Fundus = pd.read_csv(modality_df['Fundus'])
        HVF = pd.read_csv(modality_df['HVF'])
        
        rid_subjects = pd.concat([
            Clinic['RID_Subject'],
            RNFL['RID_Subject'],
            Fundus['RID_Subject'],
            HVF['RID_Subject']
        ]).drop_duplicates().reset_index(drop=True)
        sides = pd.DataFrame({'Image_Side': ['Right', 'Left']})
        expanded_subjects = rid_subjects.to_frame().merge(sides, how='cross')
        
        Clinic.drop(columns=['RID_Observation', 'Observation_ID', 'date_of_encounter_Observation'], inplace=True)
        RNFL.drop(columns=['RID_Observation', 'Observation_ID'], inplace=True)
        HVF.drop(columns=['RID_Observation', 'Observation_ID'], inplace=True)
        Fundus.drop(columns=['RID_Observation', 'Observation_ID'], inplace=True)
        multimodal_wide = pd.merge(expanded_subjects, Fundus, how='left', on=['RID_Subject'])
        multimodal_wide = pd.merge(multimodal_wide, Clinic, how='left', 
                                   on=['RID_Subject', 'Image_Side', 'Subject_ID', 'Subject_Gender', 'Subject_Ethnicity'])
        multimodal_wide = pd.merge(multimodal_wide, HVF, how='left',
                                   on=['RID_Subject', 'Subject_ID', 'Subject_Gender', 'Subject_Ethnicity', 'Image_Side'])
        multimodal_wide = pd.merge(multimodal_wide, RNFL, how='left',
                                   on=['RID_Subject', 'Subject_ID', 'Subject_Gender', 'Subject_Ethnicity', 'Image_Side'],
                                   suffixes=('_HVF', '_RNFL'))
        return multimodal_wide

    def severity_analysis(self, ds_bag: DatasetBag):
        wide = self.multimodal_wide(ds_bag)

        def compare_sides_severity(group, value_col, new_col, smaller=True): # helper method for severity_analysis
            group[new_col] = group[new_col].astype(str)
            
            if len(group) == 2:  # Ensure there are both left and right sides
                left = group[group['Image_Side'] == 'Left']
                right = group[group['Image_Side'] == 'Right']
                if not left.empty and not right.empty:
                    left_value = left[value_col].values[0]
                    right_value = right[value_col].values[0]
                    if smaller:
                        if left_value < right_value:
                            group.loc[group['Image_Side'] == 'Left', new_col] = 'Left'
                            group.loc[group['Image_Side'] == 'Right', new_col] = 'Left'
                        elif left_value == right_value:
                            group.loc[group['Image_Side'] == 'Left', new_col] = 'Left/Right'
                            group.loc[group['Image_Side'] == 'Right', new_col] = 'Left/Right'
                        else:
                            group.loc[group['Image_Side'] == 'Left', new_col] = 'Right'
                            group.loc[group['Image_Side'] == 'Right', new_col] = 'Right'
                    else:
                        # Larger value means more severe
                        if left_value > right_value:
                            group.loc[group['Image_Side'] == 'Left', new_col] = 'Left'
                            group.loc[group['Image_Side'] == 'Right', new_col] = 'Left'
                        elif left_value == right_value:
                            group.loc[group['Image_Side'] == 'Left', new_col] = 'Left/Right'
                            group.loc[group['Image_Side'] == 'Right', new_col] = 'Left/Right'
                        else:
                            group.loc[group['Image_Side'] == 'Left', new_col] = 'Right'
                            group.loc[group['Image_Side'] == 'Right', new_col] = 'Right'
            return group
        
        wide['RNFL_severe'] = np.nan
        wide = wide.groupby('RID_Subject').apply(compare_sides_severity, value_col='Average_RNFL_Thickness(μm)', new_col='RNFL_severe', smaller=True).reset_index(drop=True)
    
        wide['HVF_severe'] = np.nan
        wide = wide.groupby('RID_Subject').apply(compare_sides_severity, value_col='MD', new_col='HVF_severe', smaller=True).reset_index(drop=True)
    
        wide['CDR_severe'] = np.nan
        wide = wide.groupby('RID_Subject').apply(compare_sides_severity, value_col='CDR', new_col='CDR_severe', smaller=False).reset_index(drop=True)

        def check_severity(row):
            # "Left/Right" and "Right" should return true, and "Left/Right" and "Left" should return true, but "Left" and "Right" should return false
            # old method
            # return row['RNFL_severe'] != row['HVF_severe'] or row['RNFL_severe'] != row['CDR_severe'] or row['HVF_severe'] != row['CDR_severe']
            severities = [row['RNFL_severe'], row['HVF_severe'], row['CDR_severe']]
            try:
                return not (all(["Left" in l for l in severities]) or all(["Right" in l for l in severities]))
            except Exception: # if row is all nan
                return True
        
        wide['Severity_Mismatch'] = wide.apply(check_severity, axis=1)

        return wide

    def transform_data(self, multimodal_wide, fx_cols, y_method="all_glaucoma"):
        """
            Transforms multimodal data to create X_transformed and y as 0 and 1's; to apply to wide_train and wide_test
            Args:
                - y_method: "all_glaucoma" (Glaucoma=1, GS=0), "urgent_glaucoma" (MD<=-6 AND ICD10 diagnosis code of Glaucoma = 1, else =0)
        """

        ##### transform y and drop NA rows #####
        
        ### drop rows missing label (ie no label for POAG vs PACG vs GS)
        multimodal_wide = multimodal_wide.dropna(subset=['Label'])
        # drop rows where label is "Other" (should only be PACG, POAG, or GS)
        allowed_labels = ["PACG", "POAG", "GS"]
        multimodal_wide = multimodal_wide[multimodal_wide['Label'].isin(allowed_labels)]

        # combine PACG and POAG as glaucoma
        multimodal_wide['combined_label']= multimodal_wide['Label'].replace(['POAG', 'PACG'], 'Glaucoma')

        if y_method=="all_glaucoma":
            y = multimodal_wide.combined_label # Target variable
        elif y_method=="urgent_glaucoma":
            # drop rows missing MD
            multimodal_wide = multimodal_wide.dropna(subset=['MD'])
            multimodal_wide['MD_label'] = multimodal_wide['MD'].apply(lambda x: 'mod-severe' if x <= -6 else 'mild-GS')
            y = multimodal_wide.apply(lambda row: True if (row['combined_label'] == 'Glaucoma') and (row['MD_label'] == 'mod-severe') else False, axis=1)
        else:
            print("Not a valid y method")
        
        # convert to 0 and 1
        label_encoder = preprocessing.LabelEncoder()
        y[:] = label_encoder.fit_transform(y) # fit_transform combines fit and transform
        y = y.astype(int)

        ### transform X ###
        X = multimodal_wide[fx_cols] # Features

        ### GHT: reformat as "Outside Normal Limits", "Within Normal Limits", "Borderline", "Other"
        if "GHT" in fx_cols:
            GHT_categories = ["Outside Normal Limits", "Within Normal Limits", "Borderline"]
            X.loc[~X['GHT'].isin(GHT_categories), 'GHT'] = np.nan # alt: 'Other'; I did np.nan bc I feel like it makes more sense to drop this variable

        ### Ethnicity: reformat so that Multi-racial, Other, and ethnicity not specified are combined as Other
        if "Ethnicity" in fx_cols:
            eth_categories = ["African Descent", "Asian", "Caucasian", "Latin American"]
            X.loc[~X['Ethnicity'].isin(eth_categories), 'Ethnicity'] = 'Other'

        ### categorical data: encode using OneHotEncoder
        from feature_engine.encoding import OneHotEncoder
        categorical_vars = list(set(fx_cols) & set(['Gender', 'Ethnicity', 'GHT']))  # cateogorical vars that exist

        if len(categorical_vars)>0:
            # replace NaN with category "Unknown", then delete this column from one-hot encoding later
            for var in categorical_vars:
                X[var] = X[var].fillna("Unknown")

            encoder = OneHotEncoder(variables = categorical_vars)
            X_transformed = encoder.fit_transform(X)

            # delete Unknown columns
            X_transformed.drop(list(X_transformed.filter(regex='Unknown')), axis=1, inplace=True)

            ### sort categorical encoded columns so that they're in alphabetical order
            def sort_cols(X, var):
                # Select the subset of columns to sort
                subset_columns = [col for col in X.columns if col.startswith(var)]
                # Sort the subset of columns alphabetically
                sorted_columns = sorted(subset_columns)
                # Reorder the DataFrame based on the sorted columns
                sorted_df = X[[col for col in X.columns if col not in subset_columns] + sorted_columns]
                return sorted_df
            for var in categorical_vars:
                X_transformed = sort_cols(X_transformed, var)

        else:
            print("No categorical variables")
            X_transformed=X

        ### format numerical data
        # VFI
        if 'VFI' in fx_cols:
            X_transformed['VFI'] = X_transformed['VFI'].replace('Off', np.nan) # replace "Off" with nan
            def convert_percent(x):
                if pd.isnull(x):
                    return np.nan
                return float(x.strip('%'))/100
            X_transformed['VFI'] = X_transformed['VFI'].map(convert_percent)


        return X_transformed, y

    # current severity rule: prioritize RNFL > HVF > CDR
    # if don't want thresholds, just make threshold 0
    # just return the first eye if RNFL, MD, CDR all NaN
    def pick_severe_eye(self, df, rnfl_threshold, md_threshold):
        # Sort by 'Average_RNFL_Thickness(μm)', 'MD', and 'CDR' in descending order
        df = df.sort_values(by=['Average_RNFL_Thickness(μm)', 'MD', 'CDR'], ascending=[True, True, False])

        ### 1. if only 1 eye has a label, just pick that eye as more severe eye (for Dr. Song's patients)
        df = df.groupby('RID_Subject').apply(lambda group: group[group['Label'].notna()]).reset_index(drop=True)

        # 2. Select the row/eye with most severe value within the thresholds
        def select_row(group):
            max_value = group['Average_RNFL_Thickness(μm)'].min() # min is more severe for RNFL
            within_value_threshold = group[np.abs(group['Average_RNFL_Thickness(μm)'] - max_value) <= rnfl_threshold] # identify eyes within threshold

            if len(within_value_threshold) > 1 or len(within_value_threshold) == 0: # if both eyes "equal" RNFL OR if RNFL is NaN, then try MD
                max_other_column = within_value_threshold['MD'].min() # min is more severe for MD
                within_other_column_threshold = within_value_threshold[np.abs(within_value_threshold['MD'] - max_other_column) <= md_threshold]

                if len(within_other_column_threshold) > 1 or len(within_other_column_threshold) == 0: # if both eyes "equal" MD OR if MD is NaN, then try CDR
                    return group.sort_values(by=['CDR'], ascending=[False]).iloc[0] # since i didn't set CDR threshold, this will always pick something (even if NaN)
                else:
                    return within_other_column_threshold.iloc[0]
            else:
                return within_value_threshold.iloc[0]
        return df.groupby('RID_Subject').apply(select_row).reset_index(drop=True)

    def standardize_data(self, fx_cols, X_train, X_test):
        # identify categorical vs numeric columns
        categorical_vars = ['Gender', 'Ethnicity', 'GHT']
        numeric_vars = sorted(set(fx_cols) - set(categorical_vars), key=fx_cols.index)

        scaler = StandardScaler()

        # normalize numeric columns for X_train
        normalized_numeric_X_train = pd.DataFrame(
            scaler.fit_transform(X_train[numeric_vars]),
            columns = numeric_vars
        )
        cat_df = X_train.drop(numeric_vars, axis=1)
        X_train = pd.concat([normalized_numeric_X_train.set_index(cat_df.index), cat_df], axis=1)

        # normalize numeric columsn for X_test, but using scaler fitted to training data to prevent data leakage
        normalized_numeric_X_test = pd.DataFrame(
            scaler.transform(X_test[numeric_vars]),
            columns = numeric_vars
        )
        cat_df = X_test.drop(numeric_vars, axis=1)
        X_test = pd.concat([normalized_numeric_X_test.set_index(cat_df.index), cat_df], axis=1)

        return X_train, X_test

    # simple imputation fitted to X_train, but also applied to X_test
    def simple_impute(self, X_train_keep_missing, X_test_keep_missing, strat = "mean"):
        """
        STRATEGIES
        If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.

        If “median”, then replace missing values using the median along each column. Can only be used with numeric data.

        If “most_frequent”, then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.

        If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data.
        """
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy=strat)
        imputer = imputer.fit(X_train_keep_missing)
        X_train_imputed = imputer.transform(X_train_keep_missing)
        X_test_imputed = imputer.transform(X_test_keep_missing)
        # convert into pandas dataframe instead of np array
        X_train = pd.DataFrame(X_train_imputed, columns=X_train_keep_missing.columns)
        X_test = pd.DataFrame(X_test_imputed, columns=X_test_keep_missing.columns)

        return X_train, X_test

    # return list of pandas dataframes, each containing 1 of 10 imputations
    def mult_impute_missing(self, X, train_data=None):
        if train_data is None:
            train_data = X

        ### multiple imputation method using IterativeImputer from sklearn
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        imp = IterativeImputer(max_iter=10, random_state=0, sample_posterior=True)

        imputed_datasets = []
        for i in range(10): # 3-10 imputations standard
            imp.random_state = i
            imp.fit(train_data)
            X_imputed = imp.transform(X)
            imputed_datasets.append(pd.DataFrame(X_imputed, columns=X.columns))

        # ALTERNATIVE
        #from statsmodels.imputation import mice.MICEData # alternative package for MICE imputation
        # official docs: https://www.statsmodels.org/dev/generated/statsmodels.imputation.mice.MICE.html#statsmodels.imputation.mice.MICE
        # multiple imputation example using statsmodels: https://github.com/kshedden/mice_workshop
        #imp = mice.MICEData(data)
        #fml = 'y ~ x1 + x2 + x3 + x4' # variables used in multiple imputation model
        #mice = mice.MICE(fml, sm.OLS, imp) # OLS chosen; can change this up
        #results = mice.fit(10, 10) # 10 burn-in cycles to skip, 10 imputations
        #print(results.summary())

        return imputed_datasets

    # Logistic Regression Model Methods###
    ### 2 ways to calculate p-values; NOTE THAT P VALUES MAY NOT MAKE SENSE FOR REGULARIZED MODELS
    # https://stackoverflow.com/questions/25122999/scikit-learn-how-to-check-coefficients-significance
    @staticmethod
    def logit_pvalue(model, x):
        """ Calculate z-scores for scikit-learn LogisticRegression.
        parameters:
            model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
            x:     matrix on which the model was fit
        This function uses asymtptics for maximum likelihood estimates.
        """
        p = model.predict_proba(x)
        n = len(p)
        m = len(model.coef_[0]) + 1
        coefs = np.concatenate([model.intercept_, model.coef_[0]])
        x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
        ans = np.zeros((m, m))
        for i in range(n):
            ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
        vcov = np.linalg.inv(np.matrix(ans))
        se = np.sqrt(np.diag(vcov))
        t =  coefs/se
        p = (1 - norm.cdf(abs(t))) * 2
        return p

    @staticmethod
    def format_dec(decimals):
        func = np.vectorize(lambda x: "<.001" if x<0.001 else "%.3f"%x)
        return func(decimals)
    
    # print model coefficients, ORs, p-values
    def model_summary(self, model, X_train):
        print("Training set: %i" % len(X_train))
        coefs = model.coef_[0]
        # odd ratios = e^coef
        ors = np.exp(coefs)
        intercept = model.intercept_[0]


        p_values = self.logit_pvalue(model, X_train)

        # compare with statsmodels ### RESULT: produces same result except gives nan instead of 1.00 for insignficant p-values
        #import statsmodels.api as sm
        #sm_model = sm.Logit(y_train.reset_index(drop=True), sm.add_constant(X_train)).fit(disp=0) ### this uses y_train from outside this function so not really valid but oh well I just want it for testing purposes
        #p_values=sm_model.pvalues
        #print(self.format_dec(pvalues))
        #sm_model.summary()

        # print results
        results = pd.DataFrame({
            'Coefficient': self.format_dec(np.append(intercept, coefs)),
            'Odds Ratio': self.format_dec(np.append(np.exp(intercept), ors)),
            'P-value': self.format_dec(p_values)
        }, index=['Intercept'] + list(X_train.columns))
        print(results)
        print("")

    # model performance
    # https://medium.com/javarevisited/evaluating-the-logistic-regression-ae2decf42d61
    # helper function for compute_performance(_youden)
    @staticmethod
    def calc_stats(y_pred, y_test):
        # evaluate predictions
        mae = metrics.mean_absolute_error(y_test, y_pred)
        print('MAE: %.3f' % mae)

        # examine the class distribution of the testing set (using a Pandas Series method)
        #y_test.value_counts()
        # calculate the percentage of ones
        # because y_test only contains ones and zeros, we can simply calculate the mean = percentage of ones
        #y_test.mean()
        # calculate the percentage of zeros
        #1 - y_test.mean()

        # # Metrics computed from a confusion matrix (before thresholding)

        # Confusion matrix is used to evaluate the correctness of a classification model
        cmatrix = confusion_matrix(y_test,y_pred)

        TP = cmatrix[1, 1]
        TN = cmatrix[0, 0]
        FP = cmatrix[0, 1]
        FN = cmatrix[1, 0]

        # Classification Accuracy: Overall, how often is the classifier correct?
        # use float to perform true division, not integer division
        # print((TP + TN) / sum(map(sum, cmatrix))) -- this is is the same as the below automatic method
        print('Accuracy: %.3f' % metrics.accuracy_score(y_test, y_pred))

        # Sensitivity(recall): When the actual value is positive, how often is the prediction correct?
        sensitivity = TP / float(FN + TP)
        print('Sensitivity: %.3f' % sensitivity)
        # print('Recall score: %.3f' % metrics.recall_score(y_test, y_pred)) # same thing as sensitivity, but recall term used in ML

        # Specificity: When the actual value is negative, how often is the prediction correct?
        specificity = TN / float(TN + FP)
        print('Specificity: %.3f' % specificity)

        #from imblearn.metrics import specificity_score
        #specificity_score(y_test, y_pred)

        # Precision: When a positive value is predicted, how often is the prediction correct?
        precision = TP / float(TP + FP)
        #print('Precision: %.3f' % precision)
        print('Precision: %.3f' % metrics.precision_score(y_test, y_pred))

        # F score
        f_score = 2*TP / float(2*TP + FP + FN)
        #print('F score: %.3f' % f_score)
        print('F1 score: %.3f' % metrics.f1_score(y_test,y_pred))

        # Youden's index: = TPR - FPR = Sensitivity + Specificity - 1
        print("Calculated Youden's J index using predictions: %.3f" % (sensitivity + specificity - 1))

        #Evaluate the model using other performance metrics - REDUNDANT, COMMENTED OUT FOR NOW
        # from sklearn.metrics import classification_report
        # print(classification_report(y_test,y_pred))

        # display confusion matrix
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cmatrix, display_labels = None)
        cm_display.plot()
        plt.show()

    def compute_performance(self, model, X_test, y_test):
        print("Test set: %i" % len(X_test))
        y_pred = model.predict(X_test)

        print("-------Stats using prediction_probability of 0.5-------")
        self.calc_stats(y_pred, y_test)

    # output performance stats corresponding to OPTIMAL prediction probability cutoff per Youden's, instead of per 0.5 cutoff
    # plot_auc = True: plot individual AUC plot. If False, save to plot onto combined plot later
    def compute_performance_youden(self, model, X_test, y_test, plot=True):
        print("Model features: %s" % X_test.columns.tolist())
        # AUC
        y_pred_proba = model.predict_proba(X_test)[::,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        auc_formatted = "%.3f" % auc
        print('AUC: %s' % auc_formatted)

        # Youden's J index = sens + spec - 1 = tpr + (1-fpr) -1 = tpr - fpr
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print('Optimal prediction probability threshold by Youdens J index: %.3f' % optimal_threshold)
        youdens = tpr[optimal_idx] - fpr[optimal_idx]
        print("Optimal Youden's J index: %.3f" % youdens)
        print("Optimal Sensitivity: %.3f" % tpr[optimal_idx])
        print("Optimal Specificity: %.3f" % (1 - fpr[optimal_idx]))

        ### this is not exactly the same as the optimal numbers because it summarizes the data into predictions based on youden's optimal threshold, then computes stats based on those predictions
        #print("-------Stats using prediction_probability per YOUDEN'S-------")
        y_pred = [1 if y > optimal_threshold else 0 for y in y_pred_proba] # Predictions using optimal threshold
        #self.calc_stats(y_pred, y_test)

        if plot:
            # display confusion matrix
            cmatrix = confusion_matrix(y_test,y_pred)
            # display confusion matrix
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cmatrix, display_labels = None)
            cm_display.plot()
            plt.show()

            # ROC curve plot with optimal threshold
            plt.plot(fpr,tpr,label="AUC=%s, Youden's=%.3f" % (auc_formatted, youdens))
            plt.xlabel("False positive rate (1-specificity)")
            plt.ylabel("True positive rate (sensitivity)")
            plt.title('ROC Curve')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label='Optimal Threshold')
            plt.legend(loc=4)
            plt.show()
        return fpr, tpr, auc_formatted, optimal_idx, optimal_threshold

    #####Multiple Imputation Logistic Regression analysis methods#####
    ### After performing logistic regression on each imputed dataset, pool the results using Rubin’s rules to obtain a single set of estimates.
    # print model coefficients, ORs, p-values
    def model_summary_mice(self, logreg_models, Xtrain_finals):
        print("Training set: %i" % len(Xtrain_finals[0]))

        # Extract coefficients and standard errors
        coefs = np.array([model.coef_[0] for model in logreg_models])
        ors = np.exp(coefs)
        intercepts = np.array([model.intercept_[0] for model in logreg_models])
        p_values = np.array([logit_pvalue(model, Xtrain_finals[i]) for i, model in enumerate(logreg_models)])

        # Calculate pooled estimates
        pooled_coefs = np.mean(coefs, axis=0)
        pooled_ors = np.mean(ors, axis=0)
        pooled_intercept = np.mean(intercepts)
        # I think this calculates SES between the imputed datasets
        pooled_ses = np.sqrt(np.mean(coefs**2, axis=0) + np.var(coefs, axis=0, ddof=1) * (1 + 1/len(logreg_models)))

        pooled_p_values = np.mean(p_values, axis=0)

        # Display pooled results
        results = pd.DataFrame({
            'Coefficient': format_dec(np.append(pooled_intercept, pooled_coefs)),
            'Odds Ratio': format_dec(np.append(np.exp(pooled_intercept), pooled_ors)),
            'Standard Error': format_dec(np.append(np.nan, pooled_ses)),  # Intercept SE is not calculated here
            'P-value': format_dec(pooled_p_values)
        }, index=['Intercept'] + list(Xtrain_finals[0].columns))
        print(results)
        print("")

    # model performance
    # https://medium.com/javarevisited/evaluating-the-logistic-regression-ae2decf42d61
    def compute_performance_mice(self, logreg_models, Xtest_finals, y_test):
        print("Test set: %i" % len(Xtest_finals[0]))

        y_pred_results = []
        y_pred_proba_results = []
        for model, X_test in zip(logreg_models, Xtest_finals):
            y_pred_results.append(model.predict(X_test))

            y_pred_proba_results.append(model.predict_proba(X_test)[::,1])

        ypred_df = pd.DataFrame(np.row_stack(y_pred_results))
        y_pred = np.array(ypred_df.mode(axis=0).loc[0].astype(int)) ##### used the mode of y_pred across the 10 imputations
        y_pred_proba = np.mean(y_pred_proba_results, axis=0)


        import sklearn.metrics as metrics
        # evaluate predictions
        mae = metrics.mean_absolute_error(y_test, y_pred)
        print('MAE: %.3f' % mae)

        # examine the class distribution of the testing set (using a Pandas Series method)
        y_test.value_counts()

        # calculate the percentage of ones
        # because y_test only contains ones and zeros, we can simply calculate the mean = percentage of ones
        y_test.mean()

        # calculate the percentage of zeros
        1 - y_test.mean()

        # # Metrics computed from a confusion matrix (before thresholding)

        # Confusion matrix is used to evaluate the correctness of a classification model
        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(y_test,y_pred)
        confusion_matrix

        TP = confusion_matrix[1, 1]
        TN = confusion_matrix[0, 0]
        FP = confusion_matrix[0, 1]
        FN = confusion_matrix[1, 0]

        # Classification Accuracy: Overall, how often is the classifier correct?
        # use float to perform true division, not integer division
        # print((TP + TN) / sum(map(sum, confusion_matrix))) -- this is is the same as the below automatic method
        print('Accuracy: %.3f' % metrics.accuracy_score(y_test, y_pred))

        # Sensitivity(recall): When the actual value is positive, how often is the prediction correct?
        sensitivity = TP / float(FN + TP)

        print('Sensitivity: %.3f' % sensitivity)
        # print('Recall score: %.3f' % metrics.recall_score(y_test, y_pred)) # same thing as sensitivity, but recall term used in ML

        # Specificity: When the actual value is negative, how often is the prediction correct?
        specificity = TN / float(TN + FP)
        print('Specificity: %.3f' % specificity)

        #from imblearn.metrics import specificity_score
        #specificity_score(y_test, y_pred)

        # Precision: When a positive value is predicted, how often is the prediction correct?
        precision = TP / float(TP + FP)
        #print('Precision: %.3f' % precision)
        print('Precision: %.3f' % metrics.precision_score(y_test, y_pred))

        # F score
        f_score = 2*TP / float(2*TP + FP + FN)
        #print('F score: %.3f' % f_score)
        print('F1 score: %.3f' % metrics.f1_score(y_test,y_pred))

        #Evaluate the model using other performance metrics - REDUNDANT, COMMENTED OUT FOR NOW
        # from sklearn.metrics import classification_report
        # print(classification_report(y_test,y_pred))

        # AUC
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        print('AUC: %.3f' % auc)

        # CM matrix plot
        from sklearn import metrics
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = None)

        cm_display.plot()
        plt.show()
        # 0 = GS, 1 = POAG

        # ROC curve plot
        plt.plot(fpr,tpr,label="auc="+str(auc))
        plt.xlabel("False positive rate (1-specificity)")
        plt.ylabel("True positive rate (sensitivity)")
        plt.legend(loc=4)
        plt.show()
