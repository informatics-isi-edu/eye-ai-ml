import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePath
from typing import List, Callable, Optional

from itertools import islice
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve

from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException, FileUploadState, UploadState
from deriva_ml.dataset_bag import DatasetBag
from deriva_ml.schema_setup.system_terms import MLVocab, ExecMetadataVocab

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
                         model_version = sys.modules[globals()["__package__"]].__version__,
                         ml_schema = ml_schema)
        # self.schema = self.pb.schemas['eye-ai']
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
        diagnosis = ds_bag.get_table_as_dataframe('Diagnosis').rename(columns={'RID': 'Diagnosis_RID'}).drop(columns=['RCT', 'RMT', 'RMB'])

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
             'Diagnosis_Image', 'Cup/Disk_Ratio', 'Image_Quality']]

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
        df["Cup/Disk_Ratio"] = pd.to_numeric(df["Cup/Disk_Ratio"], errors="coerce")
        result = df.groupby("Image_RID").agg({"Cup/Disk_Ratio": cdr_func,
                                              "Diagnosis_Image": diag_func,
                                              "Image_Quality": image_quality_func})
        result = result.round({'Cup/Disk_Ratio': 4})
        result = result.fillna('NaN')
        result.reset_index('Image_RID', inplace=True)

        return result.to_dict(orient='records')

    def insert_new_diagnosis(self, pred_df: pd.DataFrame,
                             diag_tag: str,
                             execution_rid: str):
        """
        Batch insert new diagnosis entities into the Diagnosis table.

        Args:
        - pred_df (pd.DataFrame): A dataframe with column "Image" containing the image rid and "Prediction" containing 0/1.
        - diag_tag (str): Name of the diagnosis tag associated with the new entities.
        - execution_rid (str): RID of the execution which generated the diagnosis.
        """

        glaucoma = self.lookup_term("Diagnosis_Image_Vocab", "Suspected Glaucoma")
        no_glaucoma = self.lookup_term("Diagnosis_Image_Vocab", "No Glaucoma")

        mapping = {0: 'No Glaucoma', 1: 'Suspected Glaucoma'}
        pred_df['Diagnosis_Image'] = pred_df['Prediction'].map(mapping)
        pred_df = pred_df[['Image', 'Diagnosis_Image']]
        entities = pred_df.to_dict(orient='records')
        self._batch_insert(self.domain_schema.Diagnosis,
                           [{'Execution': execution_rid, 'Diagnosis_Tag': diag_tag, **e} for e in entities])

    def insert_image_annotation(self,
                                annotation_function: str,
                                annotation_type: str,
                                upload_result: dict[str, FileUploadState], metadata: pd.DataFrame) -> None:
        """
        Inserts image annotations into the catalog Image_Annotation table based on upload results and metadata.

        Parameters:
        - upload_result (str): The result of the image upload process.
        - metadata (pd.DataFrame): DataFrame containing metadata information.

        Returns:
        - None
        """
        # ToDo
        image_rids = []
        asset_rids = []

        for annotation in upload_result.values():
            if annotation.state == UploadState.success and annotation.result is not None:
                rid = annotation.result.get("RID")
                if rid is not None:
                    filename = annotation.result.get("Filename")
                    cur = metadata[metadata['Saved SVG Name'] == filename]
                    image_rid = cur['Image RID'].iloc[0]
                    image_rids.append(image_rid)
                    asset_rids.append(rid)
        annot_func_rid = self.lookup_term(table_name="Annotation_Function", term_name=annotation_function)
        annot_type_rid = self.lookup_term(table_name="Annotation_Type", term_name=annotation_type)
        self.add_attributes(image_rids,
                            asset_rids,
                            [{'Annotation_Function': annot_func_rid,
                              'Annotation_Type': annot_type_rid}] * len(image_rids)
                            )

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
        image_annot_df = ds_bag.get_table_as_dataframe('Image_Annotation')
        image_df = ds_bag.get_table_as_dataframe('Image')
        diagnosis = ds_bag.get_table_as_dataframe('Diagnosis')

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
