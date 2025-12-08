import os
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

from CapillaDiff_encoder import ConditionEncoderInference, ConditionEncoder, DictToListEncoder

DEFAULT_RELEVANT_COLUMNS = ["dysangiogenesis",
                            "enlarged_capillaries",
                            "giant_capillaries",
                            "capillary_loss",
                            "microhemorrhages",
                            "scleroderma_pattern"]


class DatasetLoader(Dataset):
    """
    HuggingFace-compatible dataset loader for CapillaDiff.
    Returns a DatasetDict with a single 'train' split, supporting:
        - shuffle()
        - select()
        - with_transform()
    Images are stored as paths (lazy loading) and captions are dicts of metadata.
    """

    def __init__(
            self, img_folder_path: str,
            metadata_csv_path: str,
            relevant_columns: list = None):
        
        self.img_folder_path = img_folder_path
        self.metadata_csv_path = metadata_csv_path
        self.relevant_columns = relevant_columns if relevant_columns is not None else DEFAULT_RELEVANT_COLUMNS

        # Load and process metadata
        self.metadata = pd.read_csv(metadata_csv_path, sep=';')
        if "FileName" not in self.metadata.columns:
            raise ValueError("metadata.csv must contain a 'FileName' column")

        # Build list of dataset items
        self.items = []
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Building HF dataset"):
            img_filename = row["FileName"]
            img_path = os.path.join(self.img_folder_path, img_filename)

            caption_dict = {
                col: row[col] if col in self.metadata.columns else None
                for col in self.relevant_columns
            }

            self.items.append({"image": img_path, "caption": caption_dict})        

        # Create HF Dataset and wrap in DatasetDict
        self.dataset_dict = DatasetDict({"train": Dataset.from_list(self.items)})
        
    def get_dataset_dict(self):
        """Return the HuggingFace DatasetDict"""
        return self.dataset_dict
    
    def get_relevant_columns(self):
        """Return the list of relevant columns used in the dataset."""
        return self.relevant_columns
    
    def get_metadata_header(self):
        """Return the first and second row of the metadata"""
        return self.metadata.columns.tolist(), self.metadata.iloc[0].tolist()
    
    def get_dataset_info(self, encoder: ConditionEncoderInference):
        """Returns a dictionary statistics about the dataset."""
        info = {}
        
        # Total number of samples
        info['total_samples'] = len(self.dataset_dict['train'])

        if encoder.get_text_mode() is None or encoder.get_text_mode() == 'simple':
            
            for col in self.relevant_columns:
                if col in self.metadata.columns:
                    # Encode the relevant columns using the provided encoder
                    # if convert_to_boolean is True, convert to boolean representation
                    # else use ranking representation
                    self.metadata[f'{col}_encoded'] = self.metadata[col].apply(encoder.get_condition)

            encoded_relevant_columns = [f"{col}_encoded" for col in self.relevant_columns]

            cond_combo = (
                self.metadata[encoded_relevant_columns + self.relevant_columns]
                .groupby(encoded_relevant_columns)
                .size()
                .reset_index(name="frequency")
                .sort_values("frequency", ascending=False)
            )

            # drop duplicates only based on encoded columns
            metadata_unique = self.metadata.drop_duplicates(subset=encoded_relevant_columns)

            # merge
            cond_combo = cond_combo.merge(
                metadata_unique[self.relevant_columns + encoded_relevant_columns],
                on=encoded_relevant_columns,
                how='left'
            )

            # add relative frequency column
            cond_combo['relative_frequency'] = cond_combo['frequency'] / info['total_samples']

            # create a name for each condition combination
            cond_combo['condition_name'] = cond_combo[encoded_relevant_columns].astype(str).agg('_'.join, axis=1)
            
            # drop the encoded columns
            cond_combo = cond_combo.drop(columns=encoded_relevant_columns)

            info['all_conditions'] = cond_combo.to_dict(orient="records")

        else:
            raise NotImplementedError("Dataset info extraction only implemented for 'None' and 'simple' text mode.")

        return info