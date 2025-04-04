import numpy as np

def gen_offsets(duration, chunk_duration, hop_duration, min_ratio=1):
    """
    duration : float, duration of the file to chunk (sec)
    chunk_duration : float, duration of the wanted chunks (sec)
    hop_duration : float, duration of the hop before rechunking (sec)
    min_ratio : float in [0, 1], manages how the end of the file is managed : if the duration is < chunk_duration*min_ratio, not added to the chunks

    returns : List[Tuple[float, float]] of all the start and end offsets
    """
    min_ = chunk_duration * min_ratio
    chunks = []
    start = 0.0

    while start < duration:
        end = min(start + chunk_duration, duration)
        start, end = np.round(start, decimals=3), np.round(end, decimals=3)
        if end - start >= min_:
            chunks.append((start, end))
        start += hop_duration

    return chunks

def overlap_ratio_dt(start_t1, end_t1, start_t2, end_t2):
    """
    start_t1, end_t1, start_t2, end_t2 : datetime objects (or datetime-like as pd.to_datetime() type)
    order doesn't matter while passing parameters as the overlap start and end are computed with max and min

    returns : float in [0, 1], the percentage of the smaller time window contain in the bigger one
    """
    overlap_start = max(start_t1, start_t2)
    overlap_end = min(end_t1, end_t2)

    duration_t1 = (end_t1 - start_t1).total_seconds()
    duration_t2 = (end_t2 - start_t2).total_seconds()
    duration_overlap = (overlap_end - overlap_start).total_seconds()

    try:
        overlap_ratio = duration_overlap / min(duration_t1, duration_t2)
        return overlap_ratio if overlap_ratio >= 0 else 0
    except:
        return 0