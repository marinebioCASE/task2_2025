import json
from ultralytics import YOLO

import preprocess_data


def predict(ds, model, conf=0.1):
    predictions_folder = ds.path_to_dataset.joinpath('predictions')
    if not predictions_folder.exists():
        if ds.images_folder.exists():
            results_list = model(source=str(ds.images_folder),
                                 project=str(ds.path_to_dataset),
                                 name=predictions_folder.name,
                                 stream=True,
                                 save=False, show=False, save_conf=True, save_txt=True, conf=conf,
                                 save_crop=False, agnostic_nms=True)

            for r in results_list:
                pass
    else:
        print('Using existing predictions folder. Delete if you wanted to overwrite')

    return predictions_folder


if __name__ == '__main__':
    path_to_dataset = input('Where is the dataset folder?')
    model_path = input('Where is the model path?')

    config_path = './dataset_config.json'
    f = open(config_path)
    config = json.load(f)

    yolo_ds = preprocess_data.YOLODataset(config, path_to_dataset)
    yolo_model = YOLO(model_path)
    output_folder = predict(yolo_ds, yolo_model)
    yolo_ds.convert_yolo_detections_to_csv(output_folder, class_encoding=config['class_encoding'])
