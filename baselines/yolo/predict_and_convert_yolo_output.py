import json

import preprocess_data


if __name__ == '__main__':
    path_to_dataset = input('Where is the dataset folder?')
    model_path =
    predictions_folder = input('Where is the predictions folder?')

    config_path = './dataset_config.json'
    f = open(config_path)
    config = json.load(f)

    ds = preprocess_data.YOLODataset(config, path_to_dataset)
    ds.convert_yolo_detections_to_csv(predictions_folder, class_encoding=config['class_encoding'])