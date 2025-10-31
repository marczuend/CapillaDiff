import os
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score, f1_score
import math
import joblib
from scipy.stats import ks_2samp
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu, levene
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis
import plotly.io as pio
from scipy.stats.stats import pearsonr, spearmanr
from scipy.stats import zscore
import itertools
import random

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, adjusted_mutual_info_score

global color_palette
color_palette = {'Stable Diffusion': 'plum', # change to gray if needed
                 'MorphoDiff': 'darkorange',
                 'StyleGan-t (class conditional)': 'magenta',
                 'StyleGan-t': 'gold',
                 'ground_truth': 'blue',
                 'Real': 'blue',
                 'Upper Bound': 'lightgray',}


def perform_correlational_analysis(df, features=None, metric='mean', result_path=None):
    """ Perform correlational analysis on the input dataframe, based on the features.
    
    Args:
    df: pandas dataframe
        The input dataframe that contains the features and metadata.
    features: list of strings
        The list of feature names to be used in the analysis.
    metric: string
        The metric to be used in the analysis. Options are 'mean', 'median', 'mode'.
    result_path: string
        The path to save the results.
        
    Returns:
    corr_of_avged_features_df: pandas dataframe
        The dataframe that contains the correlation between the average of features in groups
    """

    metadata_columns = ['Metadata_Perturbation', 'Metadata_Group', 'Metadata_Batch',
                        'Metadata_MOA', 'ImageNumber', 'PathName_input', 'FileName_input']
    features = [col for col in features if col not in metadata_columns]
    feature_df = df[features+['Metadata_Perturbation', 'Metadata_Group']]
    
    # exclude rows with Metadata_Perturbation not equal to 'dmso' and 'empty'
    feature_df = feature_df[
        (df['Metadata_Perturbation'] != 'dmso') &
        (df['Metadata_Perturbation'] != 'empty')]
        
    # check if there is any missing value in the dataframe
    missing_values = feature_df.isnull().sum().sum()
    if missing_values > 0:
        print('Number of missing values:', missing_values)
    
    perturbations = []
    for c in feature_df['Metadata_Perturbation'].unique():
        if c != 'Baseline':
            perturbations.append(c)
    print('Number of perturbations:', len(perturbations))

    groups = []
    for g in feature_df['Metadata_Group'].unique():
        if 'ground_truth' not in g.lower():
            groups.append(g)
    print('Number of groups:', len(groups))
    
    df_columns = ['Analysis', 'Metadata_Perturbation', 'Metadata_Group',
                  'Pearson_Correlation', 'Spearman_Correlation']
    corr_of_avged_features_df = pd.DataFrame(columns=df_columns)

    ground_truth_label='ground_truth'
    for group in groups:

        for perturbation in perturbations:
            
            perturbation_df = feature_df[feature_df['Metadata_Perturbation'] == perturbation]
            
            ground_truth_df = perturbation_df[perturbation_df[
                'Metadata_Group'] == ground_truth_label].reset_index(drop=True)

            if group == 'Baseline': # baseline is unconditional Stable Diffusion
                generated_df = feature_df[feature_df[
                    'Metadata_Group'] == 'Baseline'].reset_index(drop=True)
            else:
                generated_df = perturbation_df[perturbation_df[
                    'Metadata_Group'] == group].reset_index(drop=True)
            
            assert generated_df.shape[0] > 498
            assert ground_truth_df.shape[1] == generated_df.shape[1]
                
            # calculate the correlation between the average of features in group truth 
            # and the average of features in generated cohort
            assert generated_df[features].mean().shape[0] == ground_truth_df[features].shape[1]
            avg_p_correlation = 0
            if metric == 'mean':
                avg_s_correlation = ground_truth_df[features].mean().corr(
                    generated_df[features].mean(), method='spearman')

            corr_of_avged_features_df.loc[corr_of_avged_features_df.shape[0]] = \
                ['correlation of averaged features', perturbation, group,
                 avg_p_correlation, avg_s_correlation]
    
    # add two rows for the average of the correlations for each group
    for group in groups:
        corr_of_avged_features_df.loc[corr_of_avged_features_df.shape[0]] = \
            ['average', 'average', group,
             corr_of_avged_features_df[corr_of_avged_features_df['Metadata_Group'] == group][
                 'Pearson_Correlation'].mean(),
             corr_of_avged_features_df[corr_of_avged_features_df['Metadata_Group'] == group][
                 'Spearman_Correlation'].mean()]

    # calculate the p_value of the correlation between the perturbation correlations 
    # in pairs of groups except for the ground_truth group
    group_pairs = list(itertools.combinations(groups, 2))
    for pair in group_pairs:
        group1_df = corr_of_avged_features_df[
            corr_of_avged_features_df['Metadata_Group'] == pair[0]].reset_index(drop=True)
        group2_df = corr_of_avged_features_df[
            corr_of_avged_features_df['Metadata_Group'] == pair[1]].reset_index(drop=True)
        
        p_value_p = ttest_ind(group1_df['Pearson_Correlation'], group2_df['Pearson_Correlation']).pvalue
        p_value_s = ttest_ind(group1_df['Spearman_Correlation'], group2_df['Spearman_Correlation']).pvalue
        
        corr_of_avged_features_df.loc[corr_of_avged_features_df.shape[0]] = \
            ['p-value', pair[0], pair[1], p_value_p, p_value_s]
            
    
    # randomely split ground truth samples for each perturbation and calculate the correlation
    # between the average of features
    for perturbation in perturbations:
        ground_truth_df = feature_df[
            (feature_df['Metadata_Perturbation'] == perturbation) & 
            (feature_df['Metadata_Group'] == ground_truth_label)]
        
        # split the ground truth samples into two groups 100 times and calculate the correlation
        # between the average of features and select the minimum correlation
        min_p_correlation = 1
        min_s_correlation = 1
        for i in range(100):
            # set seed
            np.random.seed(i)
            random.seed(i)
            ground_truth_df1, ground_truth_df2 = train_test_split(
                ground_truth_df, test_size=0.5, random_state=i)
            if metric == 'mean':
                s_correlation = ground_truth_df1[features].mean().corr(
                    ground_truth_df2[features].mean(), method='spearman')

            if s_correlation < min_s_correlation:
                min_s_correlation = s_correlation
                print('Perturbation:', perturbation, 'Spearman:', min_s_correlation, 'seed:', i)
                
        corr_of_avged_features_df.loc[corr_of_avged_features_df.shape[0]] = \
            ['correlation of averaged features', perturbation, 'ground_truth_random_split',
                min_p_correlation, min_s_correlation]
            
    if not os.path.exists(result_path+'/feature_correlation/correlation_with_ground_truth'):
        os.makedirs(result_path+'/feature_correlation/correlation_with_ground_truth')
        
    corr_of_avged_features_df.to_csv(
        result_path+'/feature_correlation/correlation_with_ground_truth/correlation_of_feature_'+metric+'.csv',
        index=False)

    return corr_of_avged_features_df


def print_results_of_correlational_analysis(corr_of_avged_features_df, title, result_path, metric='mean'):
    """ Generate plot of the correlational analysis results.
    
    Args:
    corr_of_avged_features_df: pandas dataframe
        The dataframe that contains the correlation between the average of features in groups
    title: string
        The title of the analysis
    result_path: string
        The path to save the results
    metric: string
        The metric used in the analysis. Options are 'mean', 'median', 'mode'.
        
    """
    if not os.path.exists(result_path+'/feature_correlation/correlation_with_ground_truth'):
        os.makedirs(result_path+'/feature_correlation/correlation_with_ground_truth')

    compound_name_dict = {
        'dmso': 'DMSO',
        'empty': 'Empty',
        'Baseline': 'Baseline',
        'az138': 'AZ138',
        'az258': 'AZ258',
        'az841': 'AZ841',
        'cytochalasin-b': 'Cytochalasin B',
        'cytochalasin-d': 'Cytochalasin D',
        'latrunculin-b': 'Latrunculin B',
        'pp-2': 'PP 2',
        'demecolcine': 'Demecolcine',
        'nocodazole': 'Nocodazole',
        'colchicine': 'Colchicine',
        'vincristine': 'Vincristine',
        'epothilone-b': 'Epothilone B',
        'taxol': 'Taxol',
        'docetaxel': 'Docetaxel'}
    
    global color_palette
    plot_df = pd.DataFrame(columns=[
        'Perturbation', 'Model', 'corrplusone'])
    
    perturbation_list = corr_of_avged_features_df['Metadata_Perturbation'].unique()
    # exclude average and groups from perturbation_list
    # perturbation_list = [p for p in perturbation_list if p != 'average']
    perturbation_list = [p for p in perturbation_list if p not in corr_of_avged_features_df['Metadata_Group'].unique()]
    
    for perturbation in perturbation_list:

        assert (perturbation != 'dmso') & (perturbation != 'empty')
        sub_df = corr_of_avged_features_df[
            (corr_of_avged_features_df['Metadata_Perturbation'] == perturbation) &
            (corr_of_avged_features_df['Analysis'] == 'correlation of averaged features')].reset_index(drop=True)
         
        # get unique values of Metadata_Group
        groups = sub_df['Metadata_Group'].unique()
        for group in groups:
            corr = sub_df[sub_df['Metadata_Group'] == group]['Spearman_Correlation'].values[0]
            if group == 'Baseline':
                group = 'Stable Diffusion'
            if group == 'ground_truth_random_split':
                group = 'Upper Bound'
            plot_df.loc[
                plot_df.shape[0]] = [perturbation, group, corr+1]
            
    plot_df['Perturbation'] = plot_df['Perturbation'].map(compound_name_dict)
    # sort plot_df based on the correlation improvement
    # plot_df = plot_df.sort_values(by='Correlation Improvement', ascending=False)

    # create a new plot without specifying the x and y values
    plt.figure()
    sns.barplot(plot_df,
                x="corrplusone",
                y="Perturbation",
                hue="Model",
                palette=color_palette)
    plt.xticks(rotation=90)
    plt.xlabel('Spearman Correlation Coefficient + 1', fontsize=15)
    plt.ylabel('Perturbation', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    # add legend only to the most right figure (cytoplasm), 
    # and remove the legend from the other figures
    if 'cytoplasm' in title.lower():
        plt.legend(title='Model', title_fontsize='18', fontsize='18')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    else:
        # remove the legend
        plt.legend([],[], frameon=False)

    # plt.title(title)
    plt.savefig(
        result_path+'/feature_correlation/correlation_with_ground_truth/barplot_spearman_correlation_feature_' + \
            metric+'.png',
        format="png", bbox_inches='tight', dpi=300)
    
    return


def plot_pca(df, features, title, result_path, 
             color_metadata='Metadata_Perturbation',
             D='2D', img_type='real', style_metadata=None):
    """ Perform PCA on the input dataframe and plot the results.
    
    Args:
    df: pandas dataframe
        The input dataframe that contains the features and metadata.

    features: list of strings
        The list of feature names to be used in the analysis.

    title: string
        title of the analysis

    result_path: string
        The path to save the results
        
    color_metadata: string
        The metadata column to be used for coloring the PCA plot
    
    D: string
        The dimension of the PCA plot, either '2D' or '3D'
        
    img_type: string
        The type of the images, either 'real' or 'generated'
    
    style_metadata: string
        The metadata column to be used for styling the PCA plot
    """
    compound_name_dict = {
        'dmso': 'DMSO',
        'empty': 'DMSO',
        'Baseline': 'Baseline',
        'az138': 'AZ138',
        'az258': 'AZ258',
        'az841': 'AZ841',
        'cytochalasin-b': 'Cytochalasin B',
        'cytochalasin-d': 'Cytochalasin D',
        'latrunculin-b': 'Latrunculin B',
        'pp-2': 'PP 2',
        'demecolcine': 'Demecolcine',
        'nocodazole': 'Nocodazole',
        'colchicine': 'Colchicine',
        'vincristine': 'Vincristine',
        'epothilone-b': 'Epothilone B',
        'taxol': 'Taxol',
        'docetaxel': 'Docetaxel'}

    if not os.path.exists(result_path+'/pca'):
        os.makedirs(result_path+'/pca')
        
    feature_df = df.copy()
    # remove metadata columns from the features
    metadata_columns = ['Metadata_Perturbation', 'Metadata_Group', 'FileName_input',
                        'PathName_input', 'Metadata_Batch', 'Metadata_MOA', 'ImageNumber']
    features = [col for col in features if col not in metadata_columns]
    
    # check if feature_df[features] is empty, return
    if (feature_df[features].shape[0] == 0) | (feature_df[features].shape[1] < 2):
        return
    
    feature_df['Metadata_Perturbation'] = feature_df['Metadata_Perturbation'].map(compound_name_dict)
    collor_pallete = px.colors.qualitative.Dark24
    
    legend_title = 'Perturbation'
    if 'MOA' in color_metadata:
        legend_title = 'Mechanism of Action'
    
    ## perform PCA ##
    if D == '2D':
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(feature_df[features].values)
        feature_df['pca-one'] = pca_result[:,0]
        feature_df['pca-two'] = pca_result[:,1]
        
        # plot the clusters with color_metadata color
        plt.figure(figsize=(10,10))
        colors = ["darkorange", "dimgray", "green", "rosybrown",
                  "purple", "pink", "brown", "gold", "blue", "cyan",
                  "black", "olive", "red", "magenta"]
        
        if 'MOA' in color_metadata:
            colors = ["darkorange", "brown", "purple", "cyan", "magenta", "olive"]
        # Set your custom color palette
        customPalette = sns.set_palette(sns.color_palette(colors))
        
        # Calculate Z-scores for the columns
        feature_df['z_pca-one'] = zscore(feature_df['pca-one'])
        feature_df['z_pca-two'] = zscore(feature_df['pca-two'])

        # Define a threshold (e.g., 3 standard deviations from the mean)
        threshold = 3
        filtered_data = feature_df[(np.abs(feature_df['z_pca-one']) < threshold) & (np.abs(feature_df['z_pca-two']) < threshold)]

        if style_metadata is not None:
            
            ax = sns.scatterplot(data=filtered_data, x="pca-one", y="pca-two",
                            hue=color_metadata,
                            style=style_metadata,
                            palette=customPalette,
                            alpha=0.5,
                            s=45)
            
            style_title = 'Image Type'
            handles, labels = ax.get_legend_handles_labels()

            # Manually create the legend, specifying titles for hue and style
            l = ax.legend(handles=handles, labels=labels, title=f"{legend_title} & {style_title}", loc='best')
            l.set_title(f"{legend_title} / {style_title}")
            
        else:
            sns.scatterplot(data=filtered_data, x="pca-one", y="pca-two",
                            hue=color_metadata,
                            palette=customPalette,
                            alpha=0.5,
                            s=45)

        plt.xticks([])
        plt.yticks([])
        plt.xlabel('PC 1', fontsize=28)
        plt.ylabel('PC 2', fontsize=28)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                   title_fontsize='25',
                   fontsize='25',
                   title=legend_title)
        plt.savefig(result_path+'/pca/pca_colored_by_'+color_metadata+'_'+img_type+'_imgs_2D.png',
                        format="png", bbox_inches='tight', dpi=300)
        
        # calcualte total variance explained by the two components and write it to a file
        total_var = pca.explained_variance_ratio_.sum() * 100
        with open(result_path+'/pca/pca_'+img_type+'_imgs_variance_explained_2D.txt', 'w') as f:
            f.write('Total Explained Variance: '+str(total_var)+'%')

    elif D == '3D':
        if feature_df[features].shape[1] < 3:
            return
        
        if 'resized' in result_path:
            title = '3D PCA of CellProfiler features of standard processed images'
        elif 'cropped' in result_path:
            title = '3D PCA of CellProfiler features of cropped images'

        ## perform PCA ##
        pca = PCA(n_components=3)
        components = pca.fit_transform(feature_df[features].values)
        total_var = pca.explained_variance_ratio_.sum() * 100

        fig = px.scatter_3d(
            components, x=0, y=1, z=2, color=feature_df[color_metadata],
            title=title,
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                   title=legend_title)
        fig.update_layout(legend_title=legend_title,)
        fig.write_html(result_path+'/pca/pca_colored_by_'+color_metadata+'_'+img_type+'_imgs_3D.html')
        
        # write the total variance explained by the three components to a file
        with open(result_path+'/pca/pca_'+img_type+'_imgs_variance_explained_3D.txt', 'w') as f:
            f.write('Total Explained Variance: '+str(total_var)+'%')
    
    return


def generate_pca_plots(df, features, title, result_path):
    """ Generate PCA plots for the input dataframe real image, generated images, and all images.
    
    Args:
    df: pandas dataframe
        The input dataframe that contains the features and metadata.
        
    features: list of strings
        The list of feature names to be used in the analysis.
        
    title: string
        The title of the analysis
        
    result_path: string
        The path to save the results
    """
    perturbation_moa_dict = {
        'az138': 'Eg5 inhibitors',
        'az841': 'Aurora kinase inhibitors',
        'az258': 'Aurora kinase inhibitors',
        'cytochalasin-b': 'Actin disruptors',
        'cytochalasin-d': 'Actin disruptors',
        'latrunculin-b': 'Actin disruptors',
        'pp-2': 'Epithelial',
        'demecolcine': 'Microtubule destabilizers',
        'nocodazole': 'Microtubule destabilizers',
        'colchicine': 'Microtubule destabilizers',
        'vincristine': 'Microtubule destabilizers',
        'epothilone-b': 'Microtubule stabilizers',
        'taxol': 'Microtubule stabilizers',
        'docetaxel': 'Microtubule stabilizers'}
    
    feature_df = df.copy()
    
    # visualize PCA of real images
    real_df = feature_df[(feature_df['Metadata_Group'] == 'ground_truth') &
                         (feature_df['Metadata_Perturbation'] != 'dmso') &
                         (feature_df['Metadata_Perturbation'] != 'empty')]
    plot_pca(real_df, features, title, result_path, color_metadata='Metadata_Perturbation', D='2D', img_type='real')
    plot_pca(real_df, features, title, result_path, color_metadata='Metadata_Perturbation', D='3D', img_type='real')
    
    # visualize PCA of generated images
    generated_df = feature_df[feature_df['Metadata_Group'] == 'MorphoDiff']
    plot_pca(generated_df, features, title, result_path, color_metadata='Metadata_Perturbation',
             D='2D', img_type='generated')
    plot_pca(generated_df, features, title, result_path, color_metadata='Metadata_Perturbation',
             D='3D', img_type='generated')
    
    # visualize PCA of all images except dmso and empty
    # combine real and generated dataframes
    all_df = pd.concat([real_df, generated_df], axis=0)
    plot_pca(all_df, features, title, result_path, color_metadata='Metadata_Perturbation',
             D='2D', img_type='all', style_metadata='Metadata_Group')
    plot_pca(all_df, features, title, result_path, color_metadata='Metadata_Perturbation',
             D='3D', img_type='all', style_metadata='Metadata_Group')
    
    if ('experiment_01_resized' in result_path) | ('experiment_01_cropped' in result_path):
        
        if not 'oodtrue' in result_path.lower():

            # visualize PCA of real images colored by MOA
            real_df['Metadata_MOA'] = real_df['Metadata_Perturbation'].map(perturbation_moa_dict)
            plot_pca(real_df, features, title, result_path, color_metadata='Metadata_MOA', D='2D', img_type='real')
            plot_pca(real_df, features, title, result_path, color_metadata='Metadata_MOA', D='3D', img_type='real')
            
            # visualize PCA of generated images colored by MOA
            generated_df['Metadata_MOA'] = generated_df['Metadata_Perturbation'].map(perturbation_moa_dict)
            plot_pca(generated_df, features, title, result_path, color_metadata='Metadata_MOA',
                     D='2D', img_type='generated')
            plot_pca(generated_df, features, title, result_path, color_metadata='Metadata_MOA',
                     D='3D', img_type='generated')
            
            # visualize PCA of all images except dmso and empty colored by MOA
            all_df['Metadata_MOA'] = all_df['Metadata_Perturbation'].map(perturbation_moa_dict)
            plot_pca(all_df, features, title, result_path, color_metadata='Metadata_MOA',
                     D='2D', img_type='all', style_metadata='Metadata_Group')
            plot_pca(all_df, features, title, result_path, color_metadata='Metadata_MOA',
                     D='3D', img_type='all', style_metadata='Metadata_Group')

    return


def create_cp_correlation_matrix(df, features, title, result_path):
    """ Create a correlation matrix of the CP features between perturbations.
    
    Args:
    df: pandas dataframe
        The input dataframe that contains the features and metadata.
        
    features: list of strings
        The list of feature names to be used in the analysis.
        
    title: string
        The title of the analysis

    result_path: string
        The path to save the results
    """
    if not os.path.exists(result_path+'/feature_correlation/cp_feature_correlation_matrix'):
        os.makedirs(result_path+'/feature_correlation/cp_feature_correlation_matrix')
    
    metadata_columns = ['Metadata_Perturbation', 'Metadata_Group', 'FileName_input',
                        'PathName_input', 'Metadata_Batch', 'Metadata_MOA', 'ImageNumber']
    features = [col for col in features if col not in metadata_columns]
    
    # exclude rows with Metadata_Perturbation == 'Baseline', or 'dmso', or 'empty'
    feature_df = df[features+['Metadata_Perturbation', 'Metadata_Group']]
    feature_df = feature_df[
        (~feature_df['Metadata_Perturbation'].str.lower().str.contains('baseline')) &
        (~feature_df['Metadata_Perturbation'].str.lower().str.contains('dmso')) &
        (~feature_df['Metadata_Perturbation'].str.lower().str.contains('empty'))]

    # check if feature_df[features] is empty, return
    if (feature_df[features].shape[0] == 0):
        return
    
    perturbation_list = feature_df['Metadata_Perturbation'].unique()
    exclude_list = ['dmso', 'empty', 'Baseline']
    perturbation_list = [perturbation for perturbation in perturbation_list if perturbation not in exclude_list]
    
    group_list = feature_df['Metadata_Group'].unique()
    group_list = [group for group in group_list if group != 'Baseline']

    for group in group_list:
        mean_feature_df = pd.DataFrame(columns=features+['Metadata_Perturbation'])

        for perturbation in perturbation_list:

            sub_df = feature_df[(feature_df['Metadata_Perturbation'] == perturbation) &
                                (feature_df['Metadata_Group'] == group)]
            mean_values = sub_df[features].mean().values
            mean_feature_df.loc[mean_feature_df.shape[0]] = list(mean_values) + [perturbation]

        mean_feature_df.set_index('Metadata_Perturbation', inplace=True)

        # Transpose the DataFrame so that rows (drugs) become columns for correlation
        df_transposed = mean_feature_df.T
        
        # create correlation matrix and save it as a heatmap image
        corr = df_transposed.corr()
        plt.figure(figsize=(12,12))
        sns.heatmap(corr, cmap='coolwarm', annot=True)
        # plt.title(group)
        plt.savefig(result_path+'/feature_correlation/cp_feature_correlation_matrix/perturbation_cp_feature_correlation_matrix_'+group+'.png',
                    dpi=150, bbox_inches='tight')
    
    return


def visualize_cp_features_per_perturbation(df, features, title, result_path):
    """ Visualize CP features per perturbation in PCA plots,
    comparing distribution of DMSO, real perturbed samples and generated samples for
    each perturbation.
    
    Args:
    df: pandas dataframe
        The input dataframe that contains the features and metadata.
    
    features: list of strings
        The list of feature names to be used in the analysis.
        
    title: string
        The title of the analysis
        
    result_path: string
        The path to save the results
    """

    if not os.path.exists(result_path+'/pca/per_perturbation_dmso_real_generated_cp_visualization'):
        os.makedirs(result_path+'/pca/per_perturbation_dmso_real_generated_cp_visualization')
        
    # select rows with only dmso/empty perturbations and ground_truth as Metadata_Group
    dmso_df = df[
        (df['Metadata_Group'] == 'ground_truth') &
        (df['Metadata_Perturbation'].isin(['dmso', 'empty']))]
    
    # exclude metadata columns from the features of dmso_df
    metadata_columns = ['Metadata_Perturbation', 'Metadata_Group', 'FileName_input',
                        'PathName_input', 'Metadata_Batch', 'Metadata_MOA']
    features = [col for col in features if col not in metadata_columns]
    
    # check if the number of features is less than 2
    if dmso_df.shape[1] < 2:
        return
    dmso_df['label'] = 'DMSO'
    
    perturbation_list = df['Metadata_Perturbation'].unique()
    exclude_list = ['dmso', 'empty', 'Baseline']
    # exclude dmso and empty and Baseline from the perturbation_list
    perturbation_list = [perturbation for perturbation in perturbation_list if perturbation not in exclude_list]
    
    for perturbation in perturbation_list:
        
        perturbation_real_df = df[
            (df['Metadata_Group'] == 'ground_truth') &
            (df['Metadata_Perturbation'] == perturbation)]
        
        perturbation_generated_df = df[
            (df['Metadata_Group'] == 'MorphoDiff') &
            (df['Metadata_Perturbation'] == perturbation)]
        
        perturbation_real_df['label'] = 'Real'
        perturbation_generated_df['label'] = 'MorphoDiff'
        
        # combine dmso_df, perturbation_real_df and perturbation_generated_df
        combined_df = pd.concat(
            [dmso_df, perturbation_real_df, perturbation_generated_df], axis=0)
        
        # run PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_df[features].values)
        combined_df['pca-one'] = pca_result[:,0]
        combined_df['pca-two'] = pca_result[:,1]
        
        palette ={"DMSO": "green", "Real": "blue", "MorphoDiff": "darkorange"}
        
        plt.figure(figsize=(10,10))
        p = sns.jointplot(data=combined_df, x="pca-one", y="pca-two", hue="label", palette=palette,
                          alpha=0.6)
        p.ax_joint.set_xlabel('')
        p.ax_joint.set_ylabel('')
        p.ax_marg_x.remove()
        p.ax_marg_y.remove()

        # adjust the size of legend and remove its title
        p.ax_joint.legend(loc='upper right', title='', fontsize=15)
        p.fig.savefig(result_path+'/pca/per_perturbation_dmso_real_generated_cp_visualization/pca_'+perturbation+'.png',
                      dpi=300, bbox_inches='tight')

    return


def save_clustering_results(feature_df, features, title, result_path,
                            label='Perturbation', seed=42, result_df=None):
    """ Save the clustering results to a CSV file and evaluate the clustering performance.
    
    Args:
    feature_df: pandas dataframe
        The input dataframe that contains the features and metadata.
        
    features: list of strings
        The list of feature names to be used in the clustering.
        
    title: string
        The title of the analysis
        
    result_path: string
        The path to save the results
        
    label: string
        The metadata column to be used for clustering evaluation"""

    # External Validation Metrics (if true labels are available)
    # Assuming 'true_labels' contains the actual labels for the data
    if 'Metadata_'+label in feature_df.columns:
        true_labels = feature_df['Metadata_'+label]

        # Normalized Mutual Information
        nmi_score = normalized_mutual_info_score(true_labels, feature_df['knn_label'])

    group = feature_df['Metadata_Group'].unique()[0]
    result_df.loc[result_df.shape[0]] = [label, group, 'Normalized Mutual Information', nmi_score, seed]
    
    return result_df


def generate_barplot(result_df, metric, result_path):
    """ Generate barplot of the clustering results for each metric.
    
    Args:
    result_df: pandas dataframe
        The dataframe that contains the clustering results.
        
    metric: string
        The metric to be used in the barplot.
        
    result_path: string
        The path to save the results
        
    label: string
        The metadata column to be used for clustering evaluation
    """
    # adjust colors based on the value of Group column
    # create a new color column in result_df, where its darkorange for MorphoDiff and blue for Real, and [pink, green] for other groups randomly
    color_dic = {'MorphoDiff': 'darkorange',
                 'Real': 'blue',
                 'stylegan-t (class conditional)': 'pink',
                 'stylegan-t': 'green'}
    result_df['Color'] = result_df['Group'].map(color_dic)
    
    # filter rows with metric == metric
    result_df = result_df[result_df['Metric'] == metric]
    # create a new plot
    plt.figure(figsize=(10,10))
    sns.barplot(result_df, x="Label", y="Value", hue="Group", palette=color_dic)
    
    # remove legend title
    plt.legend(title=None,
               fontsize=15)
    if metric == 'Normalized Mutual Information':
        metric = 'NMI'
    
    # set metric as y axis label
    plt.ylabel(metric, fontsize=20)
    
    plt.xlabel(None)
    plt.xticks(fontsize=15)
    plt.ylim(0, 1)
    
    plt.savefig(result_path+"/knn_clustering/barplot_"+metric+".png",
                dpi=300)
    return


def perform_knn_cluster(df, feature, title, result_path):
    """Perform KNN clustering on the input dataframe (generated images) and 
    evaluate clustering performance.
    
    Args:
    df: pandas dataframe
        The input dataframe that contains the features and metadata.
    feature: string
        The feature name to be used in the clustering.
    title: string
        The title of the plot.
    result_path: string
        The path to save the results
    """

    if not os.path.exists(result_path+'/knn_clustering'):
        os.makedirs(result_path+'/knn_clustering')

    metadata_columns = ['Metadata_Perturbation', 'Metadata_Group', 'FileName_input',
                        'PathName_input', 'Metadata_Batch', 'Metadata_MOA', 'ImageNumber']
    features = [col for col in feature if col not in metadata_columns]
    new_df = df[features].copy()
    new_df['Metadata_Perturbation'] = df['Metadata_Perturbation']
    new_df['Metadata_Group'] = df['Metadata_Group']
    
    knn_perturbation_result_df = pd.DataFrame(columns=['Label', 'Group', 'Metric', 'Value', 'Seed'])
    knn_moa_result_df = pd.DataFrame(columns=['Label', 'Group', 'Metric', 'Value', 'Seed'])

    group_list = new_df['Metadata_Group'].unique()
    
    for group in group_list:
        
        if group == 'ground_truth':
            feature_df = new_df[
                (new_df['Metadata_Group'] == 'ground_truth') &
                (new_df['Metadata_Perturbation'] != 'dmso') &
                (new_df['Metadata_Perturbation'] != 'empty')]
            feature_df['Metadata_Group'] = 'Real'
        elif group != 'Baseline':
            feature_df = new_df[new_df['Metadata_Group'] == group]
    
        seed_list = [42, 43, 44, 45, 46]
    
        for seed in seed_list:

            # perform KMeans clustering with number of perturbations as the number of clusters
            kmeans = KMeans(n_clusters=feature_df['Metadata_Perturbation'].nunique(),
                            random_state=seed)
            kmeans.fit(feature_df[features])
            feature_df['knn_label'] = kmeans.labels_
            knn_perturbation_result_df = save_clustering_results(
                feature_df, features, title, result_path,
                label='Perturbation', seed=seed,
                result_df=knn_perturbation_result_df)
        
        # perform KMeans clustering with MOA as the number of clusters
        if ('experiment_01_resized' in result_path) | ('experiment_01_cropped' in result_path):
            if 'oodtrue' not in result_path.lower():
            
                perturbation_moa_dict = {
                    'az138': 'Eg5 inhibitors',
                    'az841': 'Aurora kinase inhibitors',
                    'az258': 'Aurora kinase inhibitors',
                    'cytochalasin-b': 'Actin disruptors',
                    'cytochalasin-d': 'Actin disruptors',
                    'latrunculin-b': 'Actin disruptors',
                    'pp-2': 'Epithelial',
                    'demecolcine': 'Microtubule destabilizers',
                    'nocodazole': 'Microtubule destabilizers',
                    'colchicine': 'Microtubule destabilizers',
                    'vincristine': 'Microtubule destabilizers',
                    'epothilone-b': 'Microtubule stabilizers',
                    'taxol': 'Microtubule stabilizers',
                    'docetaxel': 'Microtubule stabilizers'}

                feature_df['Metadata_MOA'] = feature_df['Metadata_Perturbation'].map(perturbation_moa_dict)
                
                kmeans = KMeans(n_clusters=feature_df['Metadata_MOA'].nunique(),
                                random_state=seed)
                kmeans.fit(feature_df[features])
                feature_df['knn_label'] = kmeans.labels_
                knn_moa_result_df = save_clustering_results(
                    feature_df, features, title, result_path,
                    label='MOA', seed=seed,
                    result_df=knn_moa_result_df)
                
    knn_perturbation_result_df.to_csv(result_path+'/knn_clustering/result_Perturbation_clusters.csv', index=False)
    knn_moa_result_df.to_csv(result_path+'/knn_clustering/result_MOA_clusters.csv', index=False)
    
    # combine the results of perturbation and MOA clustering
    knn_result_df = pd.concat([knn_perturbation_result_df, knn_moa_result_df], axis=0)
    ## generate barplot of the clustering results for each metric
    for metric in knn_perturbation_result_df['Metric'].unique():
        generate_barplot(knn_result_df, metric, result_path)
            
    return