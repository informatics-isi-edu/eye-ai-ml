from typing import List, Callable
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import sys


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
        self.schema = self.pb.schemas['eye-ai']

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
        subject_dataset = self.schema.Subject_Dataset
        subject = self.schema.Subject
        image = self.schema.Image
        observation = self.schema.Observation
        diagnosis = self.schema.Diagnosis
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

        image_quality_map = {e["Name"]: e["RID"] for e in self.schema.Image_Quality_Vocab.entities()}
        diagnosis_map = {e["Name"]: e["RID"] for e in self.schema.Diagnosis_Image_Vocab.entities()}
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
    
    def insert_image_annotation(self, upload_result: dict, metadata: pd.DataFrame):
        """
        Inserts image annotations into the catalog Image_Annotation table based on upload results and metadata.

        Parameters:
        - upload_result (str): The result of the image upload process.
        - metadata (pd.DataFrame): DataFrame containing metadata information.

        Returns:
        - None
        """
        image_annot_entities = []
        for annotation in upload_result.values():
            if annotation["State"] == 0 and annotation["Result"] is not None:
                rid = annotation["Result"].get("RID")
                if rid is not None:
                    filename = annotation["Result"].get("Filename")
                    cur = metadata[metadata['Saved SVG Name'] == filename]
                    image_rid = cur['Image RID'].iloc[0]
                    annot_func = cur['Worked Image Cropping Function'].iloc[0]
                    annot_func_rid = self.lookup_term(table_name="Annotation_Function", term_name=annot_func)
                    annot_type_rid = self.lookup_term(table_name="Annotation_Type", term_name="Optic Nerve")
                    image_annot_entities.append({'Annotation_Function': annot_func_rid,
                                                 'Annotation_Type': annot_type_rid,
                                                 'Image': image_rid,
                                                 'Execution_Assets': rid})
        self._batch_insert(self.schema.Image_Annotation, image_annot_entities)

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

    def create_cropped_images(self, bag_path: str, output_dir: str, crop_to_eye: bool) -> tuple:
        """
        Retrieves cropped images and saves them to the specified directory and seperated in two folders by class.

        Parameters:
        - bag_path (str): Path to the bag directory.
        - crop_to_eye (bool): Flag indicating whether to crop images to the eye.

        Returns:
        - tuple: A tuple containing the path to the directory containing cropped images and the path to the output CSV file.
        """
        cropped_path = Path(output_dir + "/Image_cropped")
        cropped_path_2SKC = Path(output_dir + "/Image_cropped/2SKC_No_Glaucoma/")
        cropped_path_2SKC.mkdir(parents=True, exist_ok=True)
        cropped_path_2SKA = Path(output_dir + "/Image_cropped/2SKA_Suspected_Glaucoma/")
        cropped_path_2SKA.mkdir(parents=True, exist_ok=True)
        svg_root_path = bag_path + '/data/assets/Image_Annotation/'
        image_root_path = bag_path + '/data/assets/Image/'
        image_annot_df = pd.read_csv(bag_path+'/data/Image_Annotation.csv')
        image_df = pd.read_csv(bag_path + '/data/Image.csv')
        diagnosis = pd.read_csv(bag_path + '/data/Diagnosis.csv')
        raw_crop = self.lookup_term(table_name="Annotation_Function", term_name='Raw_Cropped_to_Eye')

        for index, row in image_annot_df.iterrows():
            if row['Annotation_Function'] != raw_crop or crop_to_eye:
                image_rid = row['Image']
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
                image_annot_df["Cropped Filename"] = 'Cropped_' + image_file_name
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
        output_path = self.execution_assets_path/Path("ROC")
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
        'H40.05*': 'GS',
        'H40.06*': 'GS',
        'H40.10*': 'POAG', 
        'H40.11*': 'POAG', 
        'H40.12*': 'POAG', 
        'H40.15*': 'POAG', 
        'H40.2*': 'PACG',
        'H40.04*': 'Exclude',
        'H40.13*': 'Exclude',
        'H40.14*': 'Exclude',
        'H40.3*': 'Exclude',
        'H40.4*': 'Exclude',
        'H40.5*': 'Exclude',
        'H40.8*': 'Exclude',
        'H40.9*': 'Exclude' 
        }
        def map_icd_to_category(icd_code):
            for key, value in icd_mapping.items():
                if icd_code.startswith(key[:-1]):
                    return value
            return 'Other'

        # Apply the mapping
        icd10['Condition_Label'] = icd10['ICD10'].apply(map_icd_to_category)
        combined = pd.merge(icd10_asso, icd10, left_on='ICD10_Eye', right_on='RID', how='left')[['Clinical_Records', 'Condition_Label']]
        # Select severity
        priority = {'PACG': 1, 'POAG': 2, 'GS': 3, 'Exclude': 4, 'Other': 5}
        combined['Priority'] = combined['Condition_Label'].map(priority)
        combined = combined.sort_values(by=['Clinical_Records', 'Priority'])
        combined_prior = combined.drop_duplicates(subset=['Clinical_Records'], keep='first')
        combined_prior = combined_prior.drop(columns=['Priority'])
        return combined_prior[combined_prior['Condition_Label'] != 'Other']
    
    def insert_condition_label(self, condition_label: pd.DataFrame):
        label_map = {e["Name"]: e["RID"] for e in self.schema.Condition_Label.entities()}
        condition_label.replace({"Condition_Label": label_map}, inplace=True)
        condition_label.rename(columns={'Clinical_Records': 'RID'}, inplace=True)
        entities = condition_label.to_dict(orient='records')
        # return entities
        print(entities)
        self._batch_update(self.schema.Clinical_Records,
                           entities,
                           [self.schema.Clinical_Records.Condition_Label])

    def extract_modality(self, data_path):
        subject = pd.read_csv(data_path/'data/Subject.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        observation = pd.read_csv(data_path/'data/Observation.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        image = pd.read_csv(data_path/'data/Image.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        clinic = pd.read_csv(data_path/'data/Clinical_Records.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        observation_clinic_asso = pd.read_csv(data_path/'data/Observation_Clinic_Asso.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        icd10 = pd.read_csv(data_path/'data/Clinic_ICD10.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        icd10_asso = pd.read_csv(data_path/'data/Clinic_ICD_Asso.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        report = pd.read_csv(data_path/'data/Report.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        RNFL_OCR = pd.read_csv(data_path/'data/RNFL_OCR.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])
        HVF_OCR = pd.read_csv(data_path/'data/HVF_OCR.csv').drop(columns=['RCT', 'RMT', 'RCB', 'RMB'])

        
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
                              left_on= 'RID_Observation', 
                              right_on='Observation', 
                              how='left').drop(columns=['RID', 'Observation'])
        subject_obs_clinic_data = pd.merge(subject_obs_clinic, 
                                   clinic,
                                   left_on='Clinical_Records', 
                                   right_on='RID',
                                   suffixes=("_Observation", "_Clinic"),
                                   how='left').drop(columns = ['Clinical_Records']).rename(columns={'RID':'RID_Clinic'})
        # Clinical data
        clinic = pd.merge(subject_obs_clinic_data, image_side_vocab, how="left", left_on='Powerform_Laterality', right_on='Image_Side_Vocab')
        clinic = pd.merge(clinic, label_vocab, how="left", on='Condition_Label').drop(columns=['Powerform_Laterality', 'Image_Side_Vocab', 'Condition_Label'])

        # Reports
        subject_observation_report = pd.merge(subject_observation, report, 
                                      left_on='RID_Observation', 
                                      right_on='Observation', 
                                      suffixes=("subject_observation_for_HVF", "report")).drop(columns = ['Observation']).rename(columns={'RID':'RID_Report'})
        HVF = pd.merge(subject_observation_report, HVF_OCR, 
                       left_on='RID_Report', 
                       right_on='Report', 
                       suffixes=("_subject_observation_for_HVF_report", "_HVF_OCR"), 
                       how='left').rename(columns={'RID': 'RID_HVF_OCR'}).drop(columns=['Report'])
        HVF = pd.merge(HVF, image_side_vocab, how="left", on='Image_Side_Vocab').drop(columns=['Image_Side_Vocab'])
        RNFL = pd.merge(subject_observation_report, RNFL_OCR, 
                        left_on='RID_Report', 
                        right_on='Report', 
                        suffixes=("_subject_observation_for_RNFL_report", "_RNFL_OCR"), 
                        how='left').rename(columns={'RID': 'RID_RNFL_OCR'}).drop(columns=['Report'])
        RNFL = pd.merge(RNFL, image_side_vocab, how="left", on='Image_Side_Vocab').drop(columns=['Image_Side_Vocab'])
        # Image
        image = pd.merge(subject_observation, image, 
                 left_on='RID_Observation', 
                 right_on='Observation', 
                 suffixes=("_subject_observation_for_image", 
                           "_Image")).rename(columns={'RID': 'RID_RNFL_OCR'}).drop(columns=['Observation'])
        image = pd.merge(image, image_side_vocab, how="left", on='Image_Side_Vocab').drop(columns=['Image_Side_Vocab'])
        image = pd.merge(image, image_angle_vocab, how="left", on='Image_Angle_Vocab').drop(columns=['Image_Angle_Vocab'])
        # Save df
        clinic_path = PurePath(self.working_dir, 'clinic.csv')
        clinic.to_csv(clinic_path, index=False)
        HVF_path = PurePath(self.working_dir, 'HVF.csv')
        HVF.to_csv(HVF_path, index=False)
        RNFL_path = PurePath(self.working_dir, 'RNFL.csv')
        RNFL.to_csv(RNFL_path, index=False)
        image_path = PurePath(self.working_dir, 'image.csv')
        image.to_csv(image_path, index=False)
        return {"Clinic": clinic_path, "HVF": HVF_path, "RNFL": RNFL_path, "Image": image_path}

        

