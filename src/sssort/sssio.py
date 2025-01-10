import sys
import os
from pathlib import Path
import dill

import numpy as np
import neo
import quantities as pq

import logging

"""
 
 ##        #######   ######    ######   #### ##    ##  ######   
 ##       ##     ## ##    ##  ##    ##   ##  ###   ## ##    ##  
 ##       ##     ## ##        ##         ##  ####  ## ##        
 ##       ##     ## ##   #### ##   ####  ##  ## ## ## ##   #### 
 ##       ##     ## ##    ##  ##    ##   ##  ##  #### ##    ##  
 ##       ##     ## ##    ##  ##    ##   ##  ##   ### ##    ##  
 ########  #######   ######    ######   #### ##    ##  ######   
 
"""


def create_logger(filename=None, filemode='w'):
    """provide a filename to also log to file, otherwise just print"""
    # log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_fmt, datefmt=date_fmt)

    # get all loggers
    logger = logging.getLogger()

    # scope restrictions
    # logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # logging.getLogger('functions').setLevel(logging.INFO)
    disable = dict(matplotlib=logging.WARNING, functions=logging.INFO)  # to be an arg
    if disable is not None:
        for module, level in disable.items():
            logging.getLogger(module).setLevel(level)

    # for printing to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)  # <- this needs to be set as a default argument

    sys.excepthook = handle_unhandled_exception

    # config logger for writing to file
    # file_handler = logging.FileHandler(filename="%s.log" % exp_name, mode='w')
    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode=filemode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(filename=None):
    # to return a previously created logger
    logger = create_logger(filename, filemode='a')
    return logger


# logging unhandled exceptions
def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    # TODO make this cleaner that it doesn't use global namespace
    logging.critical(
        'Unhandled exception', exc_info=(exc_type, exc_value, exc_traceback)
    )


"""
 
  ######   #######  ##    ## ##     ## ######## ########  ######## ######## ########   ######  
 ##    ## ##     ## ###   ## ##     ## ##       ##     ##    ##    ##       ##     ## ##    ## 
 ##       ##     ## ####  ## ##     ## ##       ##     ##    ##    ##       ##     ## ##       
 ##       ##     ## ## ## ## ##     ## ######   ########     ##    ######   ########   ######  
 ##       ##     ## ##  ####  ##   ##  ##       ##   ##      ##    ##       ##   ##         ## 
 ##    ## ##     ## ##   ###   ## ##   ##       ##    ##     ##    ##       ##    ##  ##    ## 
  ######   #######  ##    ##    ###    ######## ##     ##    ##    ######## ##     ##  ######  
 
"""
supported_filetypes = ['.asc', '.smr', '.bin', '.csv']


def csv2seg(path, t_col=0, v_col=1, header=1):
    with open(path, 'r') as fH:
        lines = [line.strip() for line in fH.readlines()]

    times = []
    values = []
    for i, line in enumerate(lines):
        if i > header:
            t = line.split(',')[t_col]
            v = line.split(',')[v_col]
            times.append(t)
            values.append(v)
    fs = 1 / np.average(np.diff(np.array(times)))
    Asig = neo.core.AnalogSignal(
        np.array(values), units=pq.uV, sampling_rate=fs * pq.Hz
    )
    segment = neo.core.Segment()
    segment.analogsignals = [Asig]
    segment.annotate(filename=str(path))
    return segment


def asc2seg(path):
    """reads an autospike .asc file into a neo segment"""
    header_rows = 6

    with open(path, 'r') as fH:
        lines = [line.strip() for line in fH.readlines()]

    header = lines[:header_rows]
    data = lines[header_rows:]
    rec_fac = float(header[3].split(' ')[3])
    fs = float(header[4].split(' ')[3])
    Data = np.array([d.split('\t')[1] for d in data], dtype='float')
    Asig = neo.core.AnalogSignal(Data, units=pq.uV, sampling_rate=fs * pq.Hz)
    segment = neo.core.Segment()
    segment.analogsignals = [Asig]
    segment.annotate(filename=str(path))
    return segment


def raw2seg(path, fs, dtype):
    """reads a raw binary file into a neo segment. Requires manual
    specification of data type and sampling rate"""
    Data = np.fromfile(path, dtype=dtype)
    Asig = neo.core.AnalogSignal(Data, units=pq.uV, sampling_rate=fs * pq.Hz)
    segment = neo.core.Segment()
    segment.analogsignals = [Asig]
    segment.annotate(filename=str(path))
    return segment


def smr2seg(path, channel_index=0):
    """channel_index selects the respective channel in the .smr file
    that contains the voltage data"""
    channel_index = int(channel_index)
    reader = neo.io.Spike2IO(path)
    (Blk,) = reader.read(lazy=False)
    segment = Blk.segments[0]
    try:
        segment.analogsignals = [segment.analogsignals[channel_index]]
    except IndexError:
        logging.error(
            'trying to access channel with index %i in .smr file %s, but the channel is not present'
            % (channel_index, path)
        )
    segment.annotate(filename=str(path))
    return segment


"""
 
 ####  #######  
  ##  ##     ## 
  ##  ##     ## 
  ##  ##     ## 
  ##  ##     ## 
  ##  ##     ## 
 ####  #######  
 
"""


def dill2blk(path):
    with open(path, 'rb') as fH:
        Blk = dill.load(fH)
    return Blk


def blk2dill(Blk, path):
    """dumps a block via dill"""
    with open(path, 'wb') as fH:
        dill.dump(Blk, fH)


def get_data(path):
    """reads data at path"""
    ext = os.path.splitext(path)[1]
    if ext == '.dill':
        return dill2blk(path)


if __name__ == '__main__':
    """ """
    path = Path(sys.argv[1])  #
    args = sys.argv[2:] if len(sys.argv) > 1 else None

    if path.suffix in supported_filetypes:
        match path.suffix:
            case '.asc':
                seg = asc2seg(path)
            case '.csv':
                seg = csv2seg(path, *args)
            case '.smr':
                seg = smr2seg(path, *args)
            case '.bin':
                seg = raw2seg(path, *args)
    else:
        logging.critical(
            f'reading files of type {path.suffix} is currently not supported, but feel free to post an issue on github.com/grg2rsr/SSSort'
        )

    # seg = smr2seg(path, channel_index=0)
    Blk = neo.core.Block()
    Blk.segments = [seg]
    blk2dill(Blk, path.with_suffix('.dill'))
