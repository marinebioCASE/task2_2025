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


The accepted annotations are:  
'bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus'

or directly the aggregated options:
'bmabz', 'd', 'bp'

If the first one is provided, the labels will be joined following: 

'bmabz' : 'bma', 'bmb', 'bmz'
'd': 'bmd', 'bpd'
'bp': 'bp20', 'bp20plus'

To run the evaluation, the csv with the obtained detections and the mentioned format needs to be passed when asked in 
the prompt. 

The csv should follow the mentioned format. 
