# Description
This is the evaluation metrics for the bioDCASE 2025 task 2.


### Long-term data 
It's common in long-term audio analysis to segment (window) the audio. 
This challenge is an event-detection challenge. This means that you have to provide a start and end time for each 
detected event of interest (and the corresponding class annotation). 
The evaluation will then compare the overlap between the provided detections and the original annotations. A 30% overlap 
will be considered a good detection. Because the shortest baleen whale calls present in the datasets are around 1.5 
seconds, we therefore recommend to not use a time resolution lower than 5 seconds (lower time resolution = bigger window). 


### Dealing with the datetime format 
All the wav files have the following naming convention: yyyy-mm-DDTHH-MM-SS_fff.wav
This datetime of the wav file specifies the beginning of that specific file. 
Then, if a detection is found to start at second X of that file, the start_datetime of the annotations (and also the 
ones you need to prepare for results) should be the file start + X seconds.

### Which output do you need? 
The necessary output is a csv file containing these columns: 

| dataset       | filename                    | annotation | start_datetime                   | end_datetime                     |
|---------------|-----------------------------|------------|  ------------------------------- | -------------------------------- |
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bma        | 2014-02-18T21:32:03.876700+00:00 | 2014-02-18T21:32:13.281600+00:00 |
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bma        | 2014-02-18T21:37:42.187800+00:00 | 2014-02-18T21:37:51.400800+00:00 |
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bmb        | 2014-02-18T21:39:06.640300+00:00 | 2014-02-18T21:39:15.277500+00:00 |
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bmz        | 2014-02-18T21:48:19.270900+00:00 | 2014-02-18T21:48:28.292000+00:00 |


The accepted annotations names are:  
* bma
* bmb
* bmz
* bmd
* bpd
* bp20
* bp20plus

or directly the aggregated options:
* bmabz
* d
* bp

If the first one is provided, the labels will be joined following: 

bmabz : bma, bmb, bmz
d: bmd, bpd
bp: bp20, bp20plus

A combination of aggregated and not aggregated labels is also allowed as a result (and the non-aggregated ones will be 
aggregated for evaluation).

To run the evaluation, the csv with the obtained detections and the mentioned format needs to be passed when asked in 
the prompt.

The passed path can be one single csv or a folder containing all the csvs to evaluatate (both for predictions and 
ground truth).
The csv(s) should follow the mentioned format above (extra columns are not a problem if desired).

```bash
python evaluation.py 
```

## Evaluation metrics
The output is evaluated using a 1D IOU. 
IOU means intersection over union, and basically it looks at all the time when the predicted event overlaps with the 
ground truth event, divided by the total time spanning from the minimum start time to the maximum end time. 
Passive Acoustics Monitoring can be used to estimate population densities. For this reason, getting an accurate number 
of calls is crucial. Therefore, we have decided to penalize when there are several detections overlapping with one 
single ground truth. This means that if 3 predicted sound events overlap with one single ground truth event, only one of
the predicted sound events will be marked as a true positive (TP) and assigned as correct, and the rest will be marked 
as a false positive (FP).
TP are then computed counting all the prediction events which have been marked as correct. 
FP are all the prediction events which were not assigned to a ground truth. 
FN are all the ground truth events which have not been assigned any prediction.


