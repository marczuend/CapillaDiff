IMG_DIR="/cluster/customapps/medinfmk/mazuend/CapillaDiff/pseudo_data/images"
METADATA_DIR="/cluster/customapps/medinfmk/mazuend/CapillaDiff/pseudo_data/metadata.csv"

if __name__ == "__main__":
    from CapillaDiff_dataloader import CapillaDiff_datasetloader

    img_data_dir = IMG_DIR
    metadata_file_path = METADATA_DIR


    dataset = CapillaDiff_datasetloader(img_data_dir, metadata_file_path)
    #print (dataset)

    print("Dataset sample:")
    for i in range(3):
        print(dataset.dataset_dict['train'][i])