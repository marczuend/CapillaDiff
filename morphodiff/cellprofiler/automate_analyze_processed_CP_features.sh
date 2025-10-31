experiment='experiment_01_resized'
dataset='BBBC021'


preprocessing_types=(
    'standardize')

feature_types=(
    'All'
    'Cell'
    'Nuclei'
    'Cytoplasm'
    'Texture'
    'AreaShape'
    'Zernike')


for preprocessing_type in "${preprocessing_types[@]}"
do
    for feature_type in "${feature_types[@]}"
    do
        sbatch analyze_processed_CP_features.sh \
            $preprocessing_type \
            $feature_type \
            $experiment \
            $dataset
    done
done