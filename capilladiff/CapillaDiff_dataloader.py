import os
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

DEFAULT_RELEVANT_COLUMNS = ["dysangiogenesis",
                            "enlarged_capillaries",
                            "giant_capillaries",
                            "capillary_loss",
                            "microhemorrhages",
                            "scleroderma_pattern"]


class CapillaDiff_datasetloader(Dataset):
    """
    HuggingFace-compatible dataset loader for CapillaDiff.
    Returns a DatasetDict with a single 'train' split, supporting:
        - shuffle()
        - select()
        - with_transform()
    Images are stored as paths (lazy loading) and captions are dicts of metadata.
    """

    def __init__(self, img_folder_path: str, metadata_csv_path: str,
                 relevant_columns: list = None, convert_to_boolean: bool = True):
        self.img_folder_path = img_folder_path
        self.metadata_csv_path = metadata_csv_path
        self.convert_to_boolean = convert_to_boolean
        self.relevant_columns = relevant_columns if relevant_columns is not None else DEFAULT_RELEVANT_COLUMNS

        # Load and process metadata
        self.metadata = pd.read_csv(metadata_csv_path, sep=';')
        if "FileName" not in self.metadata.columns:
            raise ValueError("metadata.csv must contain a 'FileName' column")

        # Convert relevant columns to boolean if needed
        if self.convert_to_boolean:
            for col in self.relevant_columns:
                if col in self.metadata.columns:
                    self.metadata[col] = self.metadata[col].apply(
                        lambda x: 1 if isinstance(x, str) and '+' in x else 0
                    )

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