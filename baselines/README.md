# Baselines 

In each of the folders you will find all the necessary code and information to reproduce the different baselines

## ResNet

### Preprocessing

Since ResNet18 needs binary annotations, the code to preprocess original CSV files can be found in the `data_preprocess` dir.

To launch pre-processing, just do 
```python
python preprocess.py --mode <train|validation>`
```
If run as it is, new files will be saved in their original dir, with precision about the chunk and the hop length and labels information if provided.
For example, the file `casey2017_chunk5_hop2_bin_labels3.csv` is the `casey2017.csv` file with audio split into 5 seconds segments with a hop of 2 seconds (= an overlap of 3 seconds), with label binarized on their 3 values (abz, d, bp). You can of course customize saving names and pathing. 

Check the `args.py` file to see all the available options (and don't hesitate to add your own!) 

### Training
To train, you can run : 
```python
python train.py --train_annot annotations/casey2014.csv | annotations/ --val_annot annotations/casey2017.csv | annotations/
```
`train_annot` and `val_annot` options are required so the script knows where to find annotations.
If you specify a directory, all the files in will be concatenated into one big DataFrame. 

You can customize pathing and naming in the `args_train.py` file, and add your own options. 

You can run
```python
python toy_train.py --train_annot XX --val_annot YY --xp_name toy_model
```
to start a toy training. 
If pathing etc. are good, everything should run without warning or issues.
This script will create an output `~/outputs/toy_model/toy_model.pth` that will be used to test the predicting part. 

### Predicting
To predict, you can run  :
```python
python infer.py --val_annot annotations/casey2017.csv | annotations/ --modelckpt toy_model/toy_model.pth
```
`val_annot` and `modelckpt` are required so the script knows where to find annotations and the model to use to predict. 
The `modelckpt`needs the path to the model, assuming it is stored in the `outputs/` dir.

At the end of the forward pass, a CSV with columns
[dataset, filename, start_offset, $y$ (ground truth), $\hat{y}$ (predictions)]
will be created, 
in the `outputs/model_dir/`. Default name is `preds.csv`, feel free to customize it ! 

You can customize and add your own options in the `args.py` file. 

If you run
```python
python infer.py --val_annot annotations/casey2017.csv | annotations/ --modelckpt toy_model/toy_model.pth
```
a CSV file `~/outputs/toy_model/preds.csv` will be created, that will be used as example for the evaluation part. 


### Post-processing

To post-process, you can run
```python
python postprocess.py --preds_path ~/outputs/<path/to.csv>
```
`--preds_path` is require so the script knows where to find the predictions. 
It assumes predictions are stored in the `outputs/`dir. 
The CSV file is expected to have one entry per audio chunk, with the ground truth vector $y$ and the predictions vector $\hat{y}$ as columns. 

Annotations will be binarized according to to the threshold yielding the best f$_1$ score for each class and then turned into a datetime interval.

At the end, one CSV file is created in the same dir as the predictions file used as argument. It contains columns [dataset, filename, annotation, start_dt, end_dt], matching the format of the original annotation files.

Additional CSV file may be created that contains the binarized annotations if you specify a `save_name` to the `binarize_pred()` function in the `postproces.py` file.

### Evaluating
Incoming...




