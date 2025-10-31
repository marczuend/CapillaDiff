from cp_feature_analysis_functions import *
import argparse
import shutil
import os


def call_cp_analysis_functions(df, experiment, features, title, result_path):
    """ Call cellprofiler analysis functions
    
    Args:
        df (pd.DataFrame): dataframe containing the features
        experiment (str): name of the experiment
        features (list): list of features
        title (str): title of the analysis
        result_path (str): path to save the results
    """
    title = experiment+' - '+title+' - CellProfiler Features'
    
    # plot pca
    generate_pca_plots(df, features, title, result_path)

    # plot DMSO, real and generated imgs cp features for each perturbation
    visualize_cp_features_per_perturbation(df, features, title, result_path)

    # perform knn clustering with perturbation and moa labels
    perform_knn_cluster(df, features, title, result_path)

    # calculate the correlation between averaged cellprofiler features in real and generated images for each perturbation
    corr_of_avged_features_df = perform_correlational_analysis(df, features, 'mean', result_path)
    print_results_of_correlational_analysis(corr_of_avged_features_df, title, result_path, 'mean')

    # create correlation heatmap between perturbation cp features in real and generated images
    create_cp_correlation_matrix(df, features, title, result_path)
 
    return

    
def compile_list_of_features(df, feature_type):
    """ Compile a dictionary of titles and their corresponding features
    
    Args:
        df (pd.DataFrame): dataframe containing the features
        feature_type (str): string that determines which features to include in the dictionary
        
    Returns:
        list: list of features
    """
    title_feature_dict = {}

    title_feature_dict['All'] = None
    title_feature_dict['Cell'] = [col for col in df.columns if 'cell' in col.lower()]
    title_feature_dict['Nuclei'] = [col for col in df.columns if 'nuclei' in col.lower()]
    title_feature_dict['Cytoplasm'] = [col for col in df.columns if 'cytoplasm' in col.lower()]
    title_feature_dict['Texture'] = [col for col in df.columns if 'texture' in col.lower()]
    title_feature_dict['AreaShape'] = [col for col in df.columns if 'areashape' in col.lower()]
    title_feature_dict['Zernike'] = [col for col in df.columns if 'zernike' in col.lower()]

    return title_feature_dict[feature_type]


if __name__ == '__main__':
    
    result_root = 'result'
    
    # Initialize the parser
    parser = argparse.ArgumentParser(description="A simple example script with argparse.")

    # Add arguments
    parser.add_argument('--preprocessing_type',
                        type=str, help="Preprocessing type", default='standardize')
    parser.add_argument('--feature_type',
                        type=str, help="Feature type", default='All')
    parser.add_argument('--experiment',
                        type=str, help="Experiment name", default='experiment_01_resized')
    parser.add_argument('--dataset',
                        type=str, help="Dataset name", default='BBBC021')

    # Parse the arguments
    args = parser.parse_args()
                
    dataset = args.dataset
    experiment = args.experiment
    preprocessing = args.preprocessing_type
    feature_type = args.feature_type
    
    experiment_info_df = pd.DataFrame(
        columns=['Experiment', 'Preprocessing', 'FeatureFiltration', 'Title', 'Feature_count'])
    
    inputDir = f"/datasets/cellprofiler/extracted_features/{dataset}/{experiment}"
    df = pd.read_csv(f'{inputDir}/CP_features_normalized_{preprocessing}.csv')
    print(f'Loaded {df.shape[0]} images with {df.shape[1]} features')
                
    # assert Metadata_Perturbation and Metadata_Group exist in the df dataframe
    assert 'Metadata_Perturbation' in df.columns, 'Metadata_Perturbation column does not exist in the dataframe'
    assert 'Metadata_Group' in df.columns, 'Metadata_Group column does not exist in the dataframe'
                
    # fill nan values in df with mean values, excluding metadata columns
    metadata_columns = ['Metadata_Perturbation', 'Metadata_Group', 'FileName_input',
                        'PathName_input', 'ImageNumber', 'Metadata_Batch']
    features = compile_list_of_features(df, feature_type)
    if features is None:
        assert title == 'All', 'features is None, but title is not All'
        features = [col for col in df.columns if col not in metadata_columns]
    feature_df = df[features]

    # remove columns with all values as nan
    feature_df = feature_df.dropna(axis=1, how='all')
    feature_df = feature_df.fillna(feature_df.mean())
    feature_df['Metadata_Perturbation'] = df['Metadata_Perturbation']
    feature_df['Metadata_Group'] = df['Metadata_Group']
                
    # if df['Metadata_Group'].lower() contains 'morphology' or '-con', change the value to 'MorphoDiff'
    feature_df['Metadata_Group'] = feature_df['Metadata_Group'].apply(
        lambda x: 'MorphoDiff' if 'morphodiff' in x.lower() or '-con' in x.lower() else x)
    feature_df['Metadata_Group'] = feature_df['Metadata_Group'].apply(
        lambda x: 'Baseline' if 'naive' in x.lower() else x)
                
    # if df['Metadata_Perturbation'].lower() contains 'naive', change the value to Baseline
    feature_df['Metadata_Perturbation'] = feature_df['Metadata_Perturbation'].apply(
        lambda x: 'Baseline' if 'naive' in x.lower() else x)

    title = feature_type

    print(f'Analyzing {preprocessing} {title} {experiment} experiment')
    result_path = f'{result_root}/{preprocessing}/{title}/{experiment}'

    if (feature_df.shape[0] > 0) & (feature_df.shape[1] > 0):

        print('Shape of the dataframe:', feature_df[features].shape)
        experiment_info_df.loc[len(experiment_info_df)] = [
            experiment, preprocessing, feature_type, title, len(features)]
                        
        # assert that there is no nan values in the feature_df dataframe
        assert feature_df.isna().sum().sum() == 0, 'There are nan values in the dataframe'
                        
        if not os.path.exists(result_path):
            os.makedirs(result_path)
                                             
        # run analysis functions on the feature_df dataframe
        call_cp_analysis_functions(feature_df, experiment, features, title, result_path)

    else:
        print('No images to analyze')

    print(f'Finished {preprocessing} {title} {feature_type} {experiment} experiment')
    print('='*100)
            
    # check if the {result_root}/analysis/ directory exists in the result_root directory
    if not os.path.exists(f'{result_root}/analysis/'):
        os.makedirs(f'{result_root}/analysis/')

    print(experiment_info_df)
    experiment_info_df.to_csv(f'{result_root}/analysis/{dataset}_{experiment}_feature_info.csv', index=False)