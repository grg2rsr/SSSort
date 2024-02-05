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
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    # log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    date_fmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_fmt, datefmt=date_fmt)

    # get all loggers
    logger = logging.getLogger()

    # scope restrictions
    # logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # logging.getLogger('functions').setLevel(logging.INFO)
    disable = dict(matplotlib=logging.WARNING, functions=logging.INFO) # to be an arg
    if disable is not None:
        for module, level in disable.items():
            logging.getLogger(module).setLevel(level)
    
    # for printing to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO) # <- this needs to be set as a default argument
    
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
    logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

"""
 
 ######## #### ##       ########    ####  #######  
 ##        ##  ##       ##           ##  ##     ## 
 ##        ##  ##       ##           ##  ##     ## 
 ######    ##  ##       ######       ##  ##     ## 
 ##        ##  ##       ##           ##  ##     ## 
 ##        ##  ##       ##           ##  ##     ## 
 ##       #### ######## ########    ####  #######  
 
"""

def asc2seg(path):
    """ reads an autospike .asc file into a neo segment """
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
    """ reads a raw binary file into a neo segment. Requires manual
    specification of data type and sampling rate """
    Data = np.fromfile(path, dtype=dtype)
    Asig = neo.core.AnalogSignal(Data, units=pq.uV, sampling_rate=fs * pq.Hz)
    segment = neo.core.Segment()
    segment.analogsignals = [Asig]
    segment.annotate(filename=str(path))
    return segment


def smr2seg(path, channel_index=None):
    """ channel_index selects the respective channel in the .smr file
     that contains the voltage data """
    reader = neo.io.Spike2IO(path)
    Blk, = reader.read(lazy=False)
    segment = Blk.segments[0]
    if channel_index is not None:
        segment.analogsignals = [segment.analogsignals[channel_index]]
    segment.annotate(filename=str(path))
    return segment


# def list2blk(path):
#     """ convenience function for reading a file containing
#     file paths to recordings per line into a neo block """

#     with open(path, 'r') as fH:
#         fnames = [line.strip() for line in fH.readlines()]

#     Segments = []
#     for fname in fnames:
#         # logger.info("reading file %s" % fname, log=False)
#         fmt = os.path.splitext(fname)[1].lower()
#         if fmt == '.asc':
#             segment = asc2seg(fname)

#         if fmt == '.raw':
#             segment = raw2seg(fname)

#         if fmt == '.smr':
#             segment = smr2seg(fname)

#         Segments.append(segment)

#     Blk = neo.core.Block()
#     Blk.segments = Segments

#     return Blk

def dill2blk(path):
    with open(path, 'rb') as fH:
        Blk = dill.load(fH)
    return Blk


def blk2dill(Blk, path):
    """ dumps a block via dill"""
    with open(path, 'wb') as fH:
        dill.dump(Blk, fH)


def get_data(path):
    """ reads data at path """
    ext = os.path.splitext(path)[1]
    if ext == '.dill':
        return dill2blk(path)

def save_data(Blk, path):
    """ saves data to path """
    ext = os.path.splitext(path)[1]
    if ext == '.dill':
        blk2dill(Blk, path)

if __name__ == '__main__':
    """ for command line usage - first argument being path to list file """
    path = Path(sys.argv[1])
    seg = smr2seg(path, channel_index=0)
    Blk = neo.core.Block()
    Blk.segments = [seg]
    blk2dill(Blk, path.with_suffix('.dill'))

