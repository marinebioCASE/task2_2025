# task2_2025

## Which output do you need? 
The necessary output is a csv file containing these columns: 

| dataset       | filename                    | annotation | start_datetime                   | end_datetime                     |
|---------------|-----------------------------|------------|  ------------------------------- | -------------------------------- |
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bma        | 2020-02-18T21:32:03.876700+00:00 | 2020-02-18T21:32:13.281600+00:00 |
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bma        | 2020-02-18T21:37:42.187800+00:00 | 2020-02-18T21:37:51.400800+00:00 |
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bmb        | 2020-02-18T21:39:06.640300+00:00 | 2020-02-18T21:39:15.277500+00:00 |
| kerguelen2014 | 2014-02-18T21-00-00_000.wav | bmz        | 2020-02-18T21:48:19.270900+00:00 | 2020-02-18T21:48:28.292000+00:00 |


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


# Getting started 
Very often to train these models, spectrograms are used. 
This is not necessary, but it's very common.
To create the spectrograms 

