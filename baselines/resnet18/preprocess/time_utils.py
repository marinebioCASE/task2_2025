from datetime import datetime

def to_datetime(time_, format, with_zone=False):
    time_ = time_.split('.')[0] if '.wav' in time_ else time_

    if with_zone and '%z' in format:
        return datetime.strptime(time_, format)
    else:
        if '%z' in format:
            dt = datetime.strptime(time_, format)
            return dt.replace(tzinfo=None)
        else:
            return datetime.strptime(time_, format)



def to_filename(time_, format=None, with_wav=False):
    if format is None:
            if isinstance(time_, datetime):
                return time_.strftime("%Y-%m-%dT%H-%M-%S_{:03d}".format(time_.microsecond // 1000)) if not with_wav else time_.strftime("%Y-%m-%dT%H-%M-%S_{:03d}.wav".format(time_.microsecond // 1000))
            else:
                raise ValueError('format is needed')
    else:
        dt = datetime.strptime(time_, format)
        return f'{dt:%Y-%m-%dT%H-%M-%S}_{dt.microsecond // 1000:03}' if not with_wav else f'{dt:%Y-%m-%dT%H-%M-%S}_{dt.microsecond // 1000:03}.wav'



def to_iso(time_, format=None, with_zone=False):
    time_ = time_.split('.')[0] if '.wav' in time_ else time_

    if format is None:
        if isinstance(time_, datetime):
            return time_.isoformat() + '.000Z' if with_zone else time_.isoformat() + '.000'
        else:
            raise ValueError('format is needed')
    else:
        dt =  datetime.strptime(time_, format)
        return f"{dt:%Y-%m-%dT%H:%M:%S}.{dt.microsecond // 1000:03}Z" if with_zone else f"{dt:%Y-%m-%dT%H:%M:%S}.{dt.microsecond // 1000:03}"