from pathlib import Path
import pandas as pd
import pycytominer
import numpy as np
import os


def load_data(inputDir):
    """ Load cellprofiler extracted features from Image.csv files into a pandas dataframe and merge them into one dataframe
    NOTES:   
        - You should have each complete Image.csv file in a separate folder within the same directory (which will in the inputDir)
        - Each of those folders should have the name of the image group (ground_truth or the name of the folder where the generated images were)

    Args:
        inputDir (str): The directory containing the .csv files
        
    Returns:
        merged_df (pd.DataFrame): The merged dataframe containing all of the .csv files
    """
    file_paths = []
    ood_compounds = [
            'vinblastine', 'ag-1478', 'bryostatin',
            'podophyllotoxin', 'forskolin', 'dmso']
    print(os.listdir(inputDir))
    
    for perturbation in os.listdir(inputDir):
        if 'oodtrue' in inputDir:
            if (perturbation not in ood_compounds) & ('naive' not in perturbation.lower()):
                continue

        if os.path.isdir(inputDir+'/'+perturbation):
            print('Perturbation: ', perturbation)
            for group in os.listdir(inputDir+'/'+perturbation):
                # check if there is any Image.csv file in the group folder and add it to the file_paths
                if os.path.exists(inputDir+'/'+perturbation+'/'+group+'/Image.csv'):
                    file_paths.append(inputDir+'/'+perturbation+'/'+group+'/Image.csv')
                else:
                    print('Entered path: ', inputDir+'/'+perturbation+'/'+group)
                    # loop over batch folders
                    for batch in os.listdir(inputDir+'/'+perturbation+'/'+group):
                        print('Entered path: ', inputDir+'/'+perturbation+'/'+group+'/'+batch)
                        # read Image.csv files in the batch folder
                        if os.path.exists(inputDir+'/'+perturbation+'/'+group+'/'+batch+'/Image.csv'):
                            file_paths.append(inputDir+'/'+perturbation+'/'+group+'/'+batch+'/Image.csv')

    empty_feature_path = ''    
    
    # process DMSO samples of Rohban et al experiment (labelled as empty) separately       
    empty_feature_path = 'cellprofiler/extracted_features/cpg0017-rohban-pathways/experiment_04_RNA_Mito_DNA_resized/empty/ground_truth'
        
    if empty_feature_path != '':
        
        # loop over batch folders in the empty folder
        for batch in os.listdir(empty_feature_path):
            # read Image.csv files in the batch folder
            if os.path.exists(empty_feature_path+'/'+batch+'/Image.csv'):
                file_paths.append(empty_feature_path+'/'+batch+'/Image.csv')

    print('Number of .csv files: ', len(file_paths))
    
    merged_df = pd.DataFrame()

    # Load each .csv file into a pandas dataframe and merge them into one dataframe
    for path in file_paths:
        # Load .csv file into a pandas dataframe
        # check if file ends with Image.csv
        print('Path: ', path)
        if path.endswith('Image.csv'):
            df = pd.read_csv(path)
            print('Shape: ', df.shape)
            print()
            
            # check if there is not Metadata_Group column in the dataframe, add it
            if 'Metadata_Group' not in df.columns:
                group = path.split('/')[-3]
                perturbation = path.split('/')[-4]
                df['Metadata_Group'] = group
                df['Metadata_Perturbation'] = perturbation
                print('Group and Perturbation added since they were not in the Image.csv file: ', group, perturbation)
        
            # check and if any of the 'Metadata_Perturbation' and 'Metadata_Group' does not exist in df.columns, print error
            if not all(col in df.columns for col in ['Metadata_Group', 'Metadata_Perturbation']):
                print(f"Error: {path} does not have all of the required columns")
                break

            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.concat([merged_df, df], join='inner', ignore_index=True)

    print('Shape of the merged dataframe: ', merged_df.shape)
    merged_df.to_csv(inputDir+'/CP_features_not_normalized.csv', index=False)
    return merged_df


def normalize_with_pycytominer(df, normalization_method, inputDir):
    """ Normalize the merged dataframe using pycytominer and save the normalized dataframe as a .csv file
    
    Args:
        df (pd.DataFrame): The merged dataframe containing all of the .csv files
        normalization_method (str): The method to use for normalization
        inputDir (str): The directory containing the .csv files
    """

    # Explicitly convert all non-metadata columns to float
    for col in df.columns:
        if not (col in metadata_cols):
            df[col] = df[col].astype(float)

    feature_cols = df.columns.values.tolist()
    for metadata in metadata_cols:
        if metadata in feature_cols:
            feature_cols.remove(metadata)
    print('Number of feature columns: ', len(feature_cols))
    
    # This Feature selection is just for removing features with lots of NaN, so that normalization doesn't crash
    df_noNA = pycytominer.feature_select(
        profiles=df,
        features=feature_cols,   # list of features to normalize; if "infer", anything starting with 'Cells', 'Nuclei' or 'Cytoplasm'
        image_features=True,  # Whether the profiles contain image features
        operation=['drop_na_columns'],
        samples='all', # Samples to provide operation on. Str or list. Defaults to 'all'
    )
    print('Shape after removing columns with NaN values: ')
    print(df_noNA.shape)
    
    df_norm = pycytominer.normalize(
        profiles=df_noNA,    # the dataframe containing the CellProfiler features
        features=feature_cols,   # list of features to normalize; if "infer", anything starting with 'Cells', 'Nuclei' or 'Cytoplasm'
        meta_features='infer',   # list of metadata columns; if "infer" (default), any column starting with 'Metadata_'
        image_features=True,  # Whether the profiles contain image features
        samples="all",     # The metadata column values to use as a normalization reference. Defaults to "all".
        method=normalization_method,    # How to normalize the dataframe. Defaults to "standardize". "mad_robustize" is used in the profiling recipe by default
        mad_robustize_epsilon=0, # The mad_robustize fudge factor parameter. Set this to 0 if mad_robustize generates features with large values.
    )
    print('Shape after normalization: ')
    print(df_norm.shape)

    pycytominer.feature_select(
        profiles=df_norm,
        features=feature_cols,   # list of features to normalize; if "infer", anything starting with 'Cells', 'Nuclei' or 'Cytoplasm'
        image_features=False,  # Whether the profiles contain image features
        operation=['variance_threshold', 'drop_na_columns', 'blocklist', 'drop_outliers', 'correlation_threshold'],
        outlier_cutoff=500, # 500 is the default
        corr_threshold=0.9, # correlation threshold
        output_file=f"{inputDir}/CP_features_normalized_"+normalization_method+".csv", 
        samples='all', #Samples to provide operation on.Str or list. Defaults to 'all'
    )
    return


if __name__ == "__main__":

    # Create the list of metadata columns
    metadata_cols = ['FileName_input', 'PathName_input', 'Metadata_Group',
                     'Metadata_Perturbation', 'ImageNumber', 'Metadata_Batch']

    inputDir = "cellprofiler/extracted_features/BBBC021/experiment_01_resized_oodtrue"
    
    # load and merge cellprofiler extracted features into a pandas dataframe
    merged_df = load_data(inputDir)
    
    # if FileName_input and PathName_input columns exist, remove them
    if 'FileName_input' in merged_df.columns:
        merged_df = merged_df.drop('FileName_input', axis=1)
    if 'PathName_input' in merged_df.columns:
        merged_df = merged_df.drop('PathName_input', axis=1)
    if 'ImageNumber' in merged_df.columns:
        merged_df = merged_df.drop('ImageNumber', axis=1)
    if 'Metadata_Batch' in merged_df.columns:
        merged_df = merged_df.drop('Metadata_Batch', axis=1)

    # perform different normalization methods
    normalize_with_pycytominer(merged_df, 'standardize', inputDir)

