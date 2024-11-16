import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePath
from typing import List, Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve
from sklearn import preprocessing

from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException, FileUploadState, UploadState


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
                 cache_dir: str = '/data', working_dir: str = None):
        """
        Initializes the EyeAI object.

        Args:
        - hostname (str): The hostname of the server where the catalog is located.
        - catalog_number (str): The catalog number or name.
        """

        super().__init__(hostname, catalog_id, 'eye-ai',
                         cache_dir,
                         working_dir,
                         sys.modules[globals()["__package__"]].__version__)

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
            date_of_encounter = row['Date_of_Encounter']
            if subject_rid not in latest_encounters or date_of_encounter > latest_encounters[subject_rid]:
                latest_encounters[subject_rid] = date_of_encounter
        for index, row in df.iterrows():
            if row['Date_of_Encounter'] != latest_encounters[row['Subject_RID']]:
                df.drop(index, inplace=True)
        return df

    def image_tall(self, dataset_rid: str, diagnosis_tag_rid: str) -> pd.DataFrame:
        """
        Retrieve tall-format image data based on provided dataset and diagnosis tag filters.

        Args:
        - dataset_rid (str): RID of the dataset to filter images.
        - diagnosis_tag_rid (str): RID of the diagnosis tag used for further filtering.

        Returns:
        - pd.DataFrame: DataFrame containing tall-format image data from fist observation of the subject,
          based on the provided filters.
        """
        # Get references to tables to start path.
        domain_schema = self.domain_schema
        subject_dataset = domain_schema.Subject_Dataset
        subject = domain_schema.Subject
        image = domain_schema.Image
        observation = domain_schema.Observation
        diagnosis = domain_schema.Diagnosis
        path = subject_dataset.path

        results = path.filter(subject_dataset.Dataset == dataset_rid) \
            .link(subject, on=subject_dataset.Subject == subject.RID) \
            .link(observation, on=subject.RID == observation.Subject) \
            .link(image, on=observation.RID == image.Observation) \
            .filter(image.Image_Angle_Vocab == '2SK6') \
            .link(diagnosis, on=image.RID == diagnosis.Image) \
            .filter(diagnosis.Diagnosis_Tag == diagnosis_tag_rid)

        results = results.attributes(
            results.Subject.RID.alias("Subject_RID"),
            results.Observation.Date_of_Encounter,
            results.Diagnosis.RID.alias("Diagnosis_RID"),
            results.Diagnosis.RCB,
            results.Diagnosis.Image,
            results.Image.Image_Side_Vocab,
            results.Image.Filename,
            results.Diagnosis.Diagnosis_Vocab,
            results.Diagnosis.column_definitions['Cup/Disk_Ratio'],
            results.Diagnosis.Image_Quality_Vocab
        )
        image_frame = pd.DataFrame(results.fetch())

        # Select only the first observation which included in the grading app.
        image_frame = self._find_latest_observation(image_frame)

        # Show grader name
        grading_tags = ["2-35G0", "2-35RM", "2-4F74", "2-4F76"]
        diag_tag_vocab = self.list_vocabulary('Diagnosis_Tag')[["RID", "Name"]]
        if diagnosis_tag_rid in grading_tags:
            image_frame = pd.merge(image_frame, self.user_list(), how="left", left_on='RCB', right_on='ID')
        else:
            image_frame = image_frame.assign(
                Full_Name=diag_tag_vocab[diag_tag_vocab['RID'] == diagnosis_tag_rid]["Name"].item())

        # Now flatten out Diagnosis_Vocab, Image_quality_Vocab, Image_Side_Vocab
        diagnosis_vocab = self.list_vocabulary('Diagnosis_Image_Vocab')[["RID", "Name"]].rename(
            columns={"RID": 'Diagnosis_Vocab', "Name": "Diagnosis"})
        image_quality_vocab = self.list_vocabulary('Image_Quality_Vocab')[["RID", "Name"]].rename(
            columns={"RID": 'Image_Quality_Vocab', "Name": "Image_Quality"})
        image_side_vocab = self.list_vocabulary('Image_Side_Vocab')[["RID", "Name"]].rename(
            columns={"RID": 'Image_Side_Vocab', "Name": "Image_Side"})

        image_frame = pd.merge(image_frame, diagnosis_vocab, how="left", on='Diagnosis_Vocab')
        image_frame = pd.merge(image_frame, image_quality_vocab, how="left", on='Image_Quality_Vocab')
        image_frame = pd.merge(image_frame, image_side_vocab, how="left", on='Image_Side_Vocab')

        return image_frame[
            ['Subject_RID', 'Diagnosis_RID', 'Full_Name', 'Image', 'Image_Side', 'Diagnosis', 'Cup/Disk_Ratio',
             'Image_Quality']]

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
        cols = ['Image_Quality', 'Image_Side', 'Full_Name', 'Diagnosis']
        for c in cols:
            long[c] = long[c].astype('category')
        wide = pd.pivot(long, index=['Image', 'Image_Side', 'Subject_RID'], columns='Full_Name',
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

        result = df.groupby("Image").agg({"Cup/Disk_Ratio": cdr_func,
                                          "Diagnosis": diag_func,
                                          "Image_Quality": image_quality_func})
        result = result.round({'Cup/Disk_Ratio': 4})
        result = result.fillna('NaN')
        result.reset_index('Image', inplace=True)

        domain_schema = self.domain_schema
        image_quality_map = {e["Name"]: e["RID"] for e in domain_schema.Image_Quality_Vocab.entities()}
        diagnosis_map = {e["Name"]: e["RID"] for e in domain_schema.Diagnosis_Image_Vocab.entities()}
        result.replace({"Image_Quality": image_quality_map,
                        "Diagnosis": diagnosis_map}, inplace=True)
        result.rename({'Image_Quality': 'Image_Quality_Vocab', 'Diagnosis': 'Diagnosis_Vocab'}, axis=1, inplace=True)

        return result.to_dict(orient='records')

    def insert_new_diagnosis(self, pred_df: pd.DataFrame,
                             diagtag_rid: str,
                             execution_rid: str):
        """
        Batch insert new diagnosis entities into the Diagnosis table.

        Args:
        - pred_df (pd.DataFrame): A dataframe with column "Image" containing the image rid and "Prediction" containing 0/1.
        - diagtag_rid (str): RID of the diagnosis tag associated with the new entities.
        - execution_rid (str): RID of the execution which generated the diagnosis.
        """

        glaucoma = self.lookup_term("Diagnosis_Image_Vocab", "Suspected Glaucoma")
        no_glaucoma = self.lookup_term("Diagnosis_Image_Vocab", "No Glaucoma")

        mapping = {0: no_glaucoma, 1: glaucoma}
        pred_df['Diagnosis_Vocab'] = pred_df['Prediction'].map(mapping)
        pred_df = pred_df[['Image', 'Diagnosis_Vocab']]
        entities = pred_df.to_dict(orient='records')
        self._batch_insert(self.schema.Diagnosis,
                           [{'Execution': execution_rid, 'Diagnosis_Tag': diagtag_rid, **e} for e in entities])

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

    def filter_angle_2(self, bag_path: str) -> PurePath:
        """
        Filters images for just Field_2 and saves the filtered data to a CSV file.

        Parameters:
        - bag_path (str): Path to the bag directory.

        Returns:
        - str: Path to the generated CSV file containing filtered images.
        """
        Dataset_Path = PurePath(bag_path, 'data/Image.csv')
        Dataset = pd.read_csv(Dataset_Path)
        Dataset_Field_2 = Dataset[Dataset['Image_Angle_Vocab'] == "2SK6"]
        angle2_csv_path = PurePath(self.working_dir, 'Field_2.csv')
        Dataset_Field_2.to_csv(angle2_csv_path, index=False)
        return angle2_csv_path

    def get_bounding_box(self, svg_path: str) -> tuple:
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

    def create_cropped_images(self, bag_path: str, output_dir: str, crop_to_eye: bool,
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
        cropped_path = Path(output_dir + "/Image_cropped")
        cropped_path_2SKC = Path(output_dir + "/Image_cropped/2SKC_No_Glaucoma/")
        cropped_path_2SKC.mkdir(parents=True, exist_ok=True)
        cropped_path_2SKA = Path(output_dir + "/Image_cropped/2SKA_Suspected_Glaucoma/")
        cropped_path_2SKA.mkdir(parents=True, exist_ok=True)
        svg_root_path = bag_path + '/data/assets/Image_Annotation/'
        image_root_path = bag_path + '/data/assets/Image/'
        image_annot_df = pd.read_csv(bag_path + '/data/Image_Annotation.csv')
        image_df = pd.read_csv(bag_path + '/data/Image.csv')
        diagnosis = pd.read_csv(bag_path + '/data/Diagnosis.csv')
        raw_crop = self.lookup_term(table_name="Annotation_Function", term_name='Raw_Cropped_to_Eye')

        for index, row in image_annot_df.iterrows():
            if row['Annotation_Function'] != raw_crop or crop_to_eye:
                image_rid = row['Image']
                if image_rid not in exclude_list:
                    svg_path = svg_root_path + f'Cropped_{image_rid}.svg'
                    bbox = self.get_bounding_box(svg_path)
                    image_file_name = image_df[image_df['RID'] == image_rid]['Filename'].values[0]
                    image_file_path = image_root_path + image_file_name
                    image = Image.open(image_file_path)
                    cropped_image = image.crop(bbox)
                    diag = diagnosis[(diagnosis['Diagnosis_Tag'] == 'C1T4')
                                     & (diagnosis['Image'] == image_rid)]['Diagnosis_Vocab'].iloc[0]
                    if diag == '2SKC':
                        cropped_image.save(f'{str(cropped_path_2SKC)}/Cropped_{image_rid}.JPG')
                    else:
                        cropped_image.save(f'{str(cropped_path_2SKA)}/Cropped_{image_rid}.JPG')
                    image_annot_df.loc[index, 'Cropped Filename'] = 'Cropped_' + image_file_name
        output_csv = PurePath(self.working_dir, 'Cropped_Image.csv')
        image_annot_df.to_csv(output_csv)
        return cropped_path, output_csv

    def plot_roc(self, data: pd.DataFrame) -> Path:
        """
        Plot Receiver Operating Characteristic (ROC) curve based on prediction results. Save the plot values into a csv file.

        Parameters:
        - data (pd.DataFrame): DataFrame containing prediction results with columns 'True Label' and
        'Probability Score'.
        Returns:
            Path: Path to the saved csv file of ROC plot values .

        """
        output_path = self.execution_assets_path / Path("ROC")
        output_path.mkdir(parents=True, exist_ok=True)
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

    def compute_condition_label(self, icd10_asso: pd.DataFrame, icd10: pd.DataFrame) -> pd.DataFrame:
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
        icd10['Condition_Label'] = icd10['ICD10'].apply(map_icd_to_category)
        combined = pd.merge(icd10_asso, icd10, left_on='ICD10_Eye', right_on='RID', how='left')[
            ['Clinical_Records', 'Condition_Label']]
        # Select severity
        priority = {'PACG': 1, 'POAG': 2, 'GS': 3, 'Other': 4}
        combined['Priority'] = combined['Condition_Label'].map(priority)
        combined = combined.sort_values(by=['Clinical_Records', 'Priority'])
        combined_prior = combined.drop_duplicates(subset=['Clinical_Records'], keep='first')
        combined_prior = combined_prior.drop(columns=['Priority'])
        return combined_prior

    def insert_condition_label(self, condition_label: pd.DataFrame):
        label_map = {e["Name"]: e["RID"] for e in self.schema.Condition_Label.entities()}
        condition_label.replace({"Condition_Label": label_map}, inplace=True)
        condition_label.rename(columns={'Clinical_Records': 'RID'}, inplace=True)
        entities = condition_label.to_dict(orient='records')
        self._batch_update(self.schema.Clinical_Records,
                           entities,
                           [self.schema.Clinical_Records.Condition_Label])

    def extract_modality(self, data_path):
        subject = pd.read_csv(data_path / 'data/Subject.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        observation = pd.read_csv(data_path / 'data/Observation.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        image = pd.read_csv(data_path / 'data/Image.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        clinic = pd.read_csv(data_path / 'data/Clinical_Records.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        observation_clinic_asso = pd.read_csv(data_path / 'data/Observation_Clinic_Asso.csv').drop(
            columns=['RCT', 'RMT', 'RCB', 'RMB'])
        icd10 = pd.read_csv(data_path / 'data/Clinic_ICD10.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        icd10_asso = pd.read_csv(data_path / 'data/Clinic_ICD_Asso.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        report = pd.read_csv(data_path / 'data/Report.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        RNFL_OCR = pd.read_csv(data_path / 'data/RNFL_OCR.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        HVF_OCR = pd.read_csv(data_path / 'data/HVF_OCR.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])

        gender_vocab = self.list_vocabulary('Subject_Gender')[["RID", "Name"]].rename(
            columns={"RID": 'Subject_Gender', "Name": "Gender"})
        ethinicity_vocab = self.list_vocabulary('Subject_Ethnicity')[["RID", "Name"]].rename(
            columns={"RID": 'Subject_Ethnicity', "Name": "Ethnicity"})
        image_side_vocab = self.list_vocabulary('Image_Side_Vocab')[["RID", "Name"]].rename(
            columns={"RID": 'Image_Side_Vocab', "Name": "Side"})
        image_angle_vocab = self.list_vocabulary('Image_Angle_Vocab')[["RID", "Name"]].rename(
            columns={"RID": 'Image_Angle_Vocab', "Name": "Angle"})
        label_vocab = self.list_vocabulary('Condition_Label')[["RID", "Name"]].rename(
            columns={"RID": 'Condition_Label', "Name": "Label"})

        subject = pd.merge(subject, gender_vocab, how="left", on='Subject_Gender')
        subject = pd.merge(subject, ethinicity_vocab, how="left", on='Subject_Ethnicity')
        subject = subject[['RID', 'Subject_ID', 'Gender', 'Ethnicity']]

        subject_observation = pd.merge(subject, observation, left_on='RID', right_on='Subject', how='left',
                                       suffixes=('_Subject', '_Observation')).drop(columns=['Subject'])
        subject_obs_clinic = pd.merge(subject_observation,
                                      observation_clinic_asso,
                                      left_on='RID_Observation',
                                      right_on='Observation',
                                      how='left').drop(columns=['RID', 'Observation'])
        subject_obs_clinic_data = pd.merge(subject_obs_clinic,
                                           clinic,
                                           left_on='Clinical_Records',
                                           right_on='RID',
                                           suffixes=("_Observation",""),
                                           how='left').drop(columns=['Clinical_Records']).rename(
            columns={'RID': 'RID_Clinic'})
        # Clinical data
        clinic = pd.merge(subject_obs_clinic_data, image_side_vocab, how="left", left_on='Powerform_Laterality',
                          right_on='Image_Side_Vocab')
        clinic = pd.merge(clinic, label_vocab, how="left", on='Condition_Label').drop(
            columns=['Powerform_Laterality', 'Image_Side_Vocab', 'Condition_Label'])

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
                       how='left').rename(columns={'RID': 'RID_HVF_OCR'}).drop(columns=['Report'])
        HVF = pd.merge(HVF, image_side_vocab, how="left", on='Image_Side_Vocab').drop(columns=['Image_Side_Vocab'])

        def select_24_2(HVF):
            HVF_clean = HVF.dropna(subset=['RID_HVF_OCR'])
            priority = {'24-2': 1, '10-2': 2, '30-2': 3}
            HVF_clean['priority'] = HVF_clean['Field_Size'].map(priority)
            HVF_sorted = HVF_clean.sort_values(by=['RID_Observation', 'priority'])
            result = HVF_sorted.groupby(['RID_Observation', 'Side']).first().reset_index()
            result = result.drop(columns=['priority'])
            return result

        HVF = select_24_2(HVF)

        # Report_RNFL
        RNFL = pd.merge(subject_observation_report, RNFL_OCR,
                        left_on='RID_Report',
                        right_on='Report',
                        suffixes=("_subject_observation_for_RNFL_report", "_RNFL_OCR"),
                        how='left').rename(columns={'RID': 'RID_RNFL_OCR'}).drop(columns=['Report'])
        RNFL = pd.merge(RNFL, image_side_vocab, how="left", on='Image_Side_Vocab').drop(columns=['Image_Side_Vocab'])

        def highest_signal_strength(RNFL):
            RNFL_clean = RNFL.dropna(subset=['RID_RNFL_OCR', 'Signal_Strength'])
            idx = RNFL_clean.groupby(['RID_Observation', 'Side'])['Signal_Strength'].idxmax()
            result = RNFL_clean.loc[idx]
            return result

        RNFL = highest_signal_strength(RNFL)
        # Image
        image = pd.merge(subject_observation, image,
                         left_on='RID_Observation',
                         right_on='Observation',
                         suffixes=("_subject_observation_for_image",
                                   "_Image")).rename(columns={'RID': 'RID_Image'}).drop(columns=['Observation'])
        image = pd.merge(image, image_side_vocab, how="left", on='Image_Side_Vocab').drop(columns=['Image_Side_Vocab'])
        image = pd.merge(image, image_angle_vocab, how="left", on='Image_Angle_Vocab').drop(
            columns=['Image_Angle_Vocab'])

        # Select the observation according fundus date of encounter
        fundus = image[['RID_Subject', 'Subject_ID', 'Gender', 'Ethnicity', 'RID_Observation', 'Observation_ID',
                        'Date_of_Encounter']].drop_duplicates()

        def closest_to_fundus(report, fundus):
            report['Date_of_Encounter'] = pd.to_datetime(report['Date_of_Encounter']).dt.tz_localize(None)
            fundus['Date_of_Encounter'] = pd.to_datetime(fundus['Date_of_Encounter']).dt.tz_localize(None)
            report_match = pd.DataFrame()

            def find_closest_date(target_date, dates):
                return min(dates, key=lambda d: abs(d - target_date))

            for idx, row in fundus.iterrows():
                rid = row['RID_Subject']
                target_date = row['Date_of_Encounter']

                for side in ['Left', 'Right']:
                    filtered_data = report[(report['RID_Subject'] == rid) & (report['Side'] == side)]
                    if not filtered_data.empty:
                        # Find the closest date entry
                        if sum(filtered_data['Date_of_Encounter'].isna()) > 0:
                            report_match = pd.concat([report_match, filtered_data.iloc[[0]]])
                        else:
                            closest_date = find_closest_date(target_date, filtered_data['Date_of_Encounter'])
                            closest_entries = filtered_data[filtered_data['Date_of_Encounter'] == closest_date]
                            report_match = pd.concat([report_match, closest_entries])
            return report_match

        HVF_match = closest_to_fundus(HVF, fundus)
        RNFL_match = closest_to_fundus(RNFL, fundus)
        
        # select clinic records by the date of encounter (on the fundus date of encounter)
        # results in 2078 records from 1062 subjects
        clinic_match = pd.merge(fundus, clinic, how='left', on='RID_Observation', suffixes=("", "_Clinic"))[
            ['RID_Subject', 'Subject_ID', 'Gender', 'Ethnicity', 'RID_Observation',
             'Observation_ID', 'Date_of_Encounter_Observation', 'RID_Clinic',
             'Date_of_Encounter_Clinic', 'LogMAR_VA', 'Visual_Acuity_Numerator', 'IOP',
             'Refractive_Error', 'CCT', 'CDR', 'Gonioscopy', 'Condition_Display', 'Provider',
             'Clinical_ID', 'Side', 'Label']]

        RNFL_match.rename(columns={'Date_of_Encounter': 'Date_of_Encounter_RNFL'}, inplace=True)
        HVF_match.rename(columns={'Date_of_Encounter': 'Date_of_Encounter_HVF'}, inplace=True)
        fundus.rename(columns={'Date_of_Encounter': 'Date_of_Encounter_Fundus'}, inplace=True)

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

    def multimodal_wide(self, data_path):
        modality_df = self.extract_modality(data_path)
        Clinic = pd.read_csv(modality_df['Clinic'])
        RNFL = pd.read_csv(modality_df['RNFL'])
        Fundus = pd.read_csv(modality_df['Fundus'])
        HVF = pd.read_csv(modality_df['HVF'])
        
        rid_subjects = pd.concat([
            Clinic['RID_Subject'],
            RNFL['RID_Subject'],
            Fundus['RID_Subject'],
            HVF['RID_Subject']
        ]).drop_duplicates().reset_index(drop=True)
        sides = pd.DataFrame({'Side': ['Right', 'Left']})
        expanded_subjects = rid_subjects.to_frame().merge(sides, how='cross')
        
        Clinic.drop(columns=['RID_Observation', 'Observation_ID', 'Date_of_Encounter_Observation'], inplace=True)
        RNFL.drop(columns=['RID_Observation', 'Observation_ID'], inplace=True)
        HVF.drop(columns=['RID_Observation', 'Observation_ID'], inplace=True)
        Fundus.drop(columns=['RID_Observation', 'Observation_ID'], inplace=True)
        multimodal_wide = pd.merge(expanded_subjects, Fundus, how='left', on=['RID_Subject'])
        multimodal_wide = pd.merge(multimodal_wide, Clinic, how='left', 
                                   on=['RID_Subject', 'Side', 'Subject_ID', 'Gender', 'Ethnicity'])
        multimodal_wide = pd.merge(multimodal_wide, HVF, how='left',
                                   on=['RID_Subject', 'Subject_ID', 'Gender', 'Ethnicity', 'Side'])
        multimodal_wide = pd.merge(multimodal_wide, RNFL, how='left',
                                   on=['RID_Subject', 'Subject_ID', 'Gender', 'Ethnicity', 'Side'],
                                   suffixes=('_HVF', '_RNFL'))
        return multimodal_wide

    def severity_analysis(self, data_path):
        wide = self.multimodal_wide(data_path)

        def compare_sides_severity(group, value_col, new_col, smaller=True): # helper method for severity_analysis
            group[new_col] = group[new_col].astype(str)
            
            if len(group) == 2:  # Ensure there are both left and right sides
                left = group[group['Side'] == 'Left']
                right = group[group['Side'] == 'Right']
                if not left.empty and not right.empty:
                    left_value = left[value_col].values[0]
                    right_value = right[value_col].values[0]
                    if smaller:
                        if left_value < right_value:
                            group.loc[group['Side'] == 'Left', new_col] = 'Left'
                            group.loc[group['Side'] == 'Right', new_col] = 'Left'
                        elif left_value == right_value:
                            group.loc[group['Side'] == 'Left', new_col] = 'Left/Right'
                            group.loc[group['Side'] == 'Right', new_col] = 'Left/Right'
                        else:
                            group.loc[group['Side'] == 'Left', new_col] = 'Right'
                            group.loc[group['Side'] == 'Right', new_col] = 'Right'
                    else:
                        # Larger value means more severe
                        if left_value > right_value:
                            group.loc[group['Side'] == 'Left', new_col] = 'Left'
                            group.loc[group['Side'] == 'Right', new_col] = 'Left'
                        elif left_value == right_value:
                            group.loc[group['Side'] == 'Left', new_col] = 'Left/Right'
                            group.loc[group['Side'] == 'Right', new_col] = 'Left/Right'
                        else:
                            group.loc[group['Side'] == 'Left', new_col] = 'Right'
                            group.loc[group['Side'] == 'Right', new_col] = 'Right'
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
            - y_method: "all_glaucoma" (Glaucoma=1, GS=0), "urgent_glaucoma" (MD<=-6 = 1, GS=0)
    """
        ### drop rows missing label (ie no label for POAG vs PACG vs GS)
        multimodal_wide = multimodal_wide.dropna(subset=['Label'])
        # drop rows where label is "Other" (should only be PACG, POAG, or GS)
        allowed_labels = ["PACG", "POAG", "GS"]
        multimodal_wide = multimodal_wide[multimodal_wide['Label'].isin(allowed_labels)]
    
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
    
        ### transform y
        if y_method=="all_glaucoma":
            y = multimodal_wide.Label # Target variable
            # combine PACG and POAG as glaucoma
            y = y.replace(['POAG', 'PACG'], 'Glaucoma')
        elif y_method=="urgent_glaucoma":
            y = multimodal_wide['MD'].apply(lambda x: 'mod-severe' if x <= -6 else 'mild-GS')
        else:
            print("Not a valid y method")
        # convert to 0 and 1
        label_encoder = preprocessing.LabelEncoder()
        y[:] = label_encoder.fit_transform(y) # fit_transform combines fit and transform
        y = y.astype(int)
    
        return X_transformed, y
