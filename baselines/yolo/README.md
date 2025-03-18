# Training a YOLOv11 object detector

YOLOv11 is an object detector model which can be re-trained on custom data. 
To do so, first the challenge data needs to be converted to the YOLO format. 
To this aim, first run the preprocess_data.py script.

## Create the environment using poetry
You will need to install poetry first. 
```bash
poetry install
```

## Preprocess dataset
First you will need to adjust the dataset_config.json parameters if desired. 
The options are to modify the spectrogram parameters: 

```json 
{
"duration" : 30,
"overlap" : 0.5,
"desired_fs" : 250,
"channel" : 0,
"log": false,
"color": false,

"nfft" : 512,
"win_len" : 256,
"hop_len" : 20,

"class_encoding": {"bma": 0,"bmb": 1, "bmz": 2, "bmd": 3, "bpd": 4, "bp20": 5, "bp20plus": 6}
}
```

Then you will have to run
1. Run the preprocess_data.py script and pass the folder to the training folder
2. Run the preprocess_data.py script and pass the folder to the validation folder 
3. Run the preprocess_data.py script and pass the folder to the test folder


## Train the YOLO model
First, adjust the paths on the custom.yaml file to point to the train and valid folders.
Then, run the train_yolo.py script 
```bash
python train_yolo.py 
```
This will generate the automatic YOLO output in the folder specified in the "path" argument of the yaml config file. 
It will also automatically run the YOLO predictions on the test set.

## Convert the YOLO output to the required format
To be able to compare the YOLO results with the ground truth annotations we will need to re-process the obtained results. 
To do that, run the convert_yolo_output.py script 

```bash 
python convert_yolo_output.py 
```