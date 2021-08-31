import numpy as np
import sys
import math

### Auxiliary funcs 

def pow2_ceil(x):
    x = int(x)
    if (x < 0):
        return 0
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1

def dec2bit(value, bits=8):
    """ bits=8: 42 -> (False, True, False, True, False, True, False, False) """
    v = value % 2**bits
    seq = tuple(True if c == '1' else False for c in bin(v)[2:].zfill(bits)[::-1])
    if value - v > 0: seq = seq + dec2bit(value // 2**bits)
    return seq


### Read Config params

def parseCFG(f,tx_ant,rx_ant):
    txMask, rxMask =  compChannel(tx_ant,rx_ant)
    cfg = {"profileCfg":{},
            "frameCfg":{},
            "channelCfg":{"rxMask":rxMask,"txMask":txMask}} 
    for line in f:
        if "profileCfg" in line:
            params = [float(el) for el in line[11::].split()]
            cfg["profileCfg"] = {
                "id": params[0],
                "startFreq": params[1],
                "idleTime": params[2],
                "startTime": params[3],
                "rampEndTime":params[3],
                "txOutPower": params[5],
                "txPhaseShifter":params[6],
                "freqSlope": params[7],
                "txStartTime": params[8],
                "adcSamples": params[9],
                "sampleRate": params[10],
                "hpfCornerFreq1": params[11],
                "hpfCornerFreq2": params[12],
                "rxGain": params[13]}
        elif "frameCfg" in line:
            params = [int(el) for el in line[8::].split()]
            cfg["frameCfg"] = {
                "startIndex": params[0],
                "endIndex":params[1],
                "loops": params[2],
                "frames":params[3],
                "periodicity": params[4],
                "trigger": params[5],
                "triggerDelay": params[6]}
    return cfg

def compChannel(tx_ant,rx_ant):
    rxMask = 2 ** rx_ant - 1
    n = tx_ant
    if n==1:
        n = 0
    else:
        n = 2 * n
    txMask = 1 + n
    return txMask, rxMask


# Compute settings

def num_tx_antenna(cfg, mask=(True, True, True)):
    b = dec2bit(cfg['channelCfg']['txMask'], 3)
    m = (True,) * (len(b) - len(mask)) + mask
    res = [digit if valid else 0 for digit, valid in zip(b, m)]
    return sum(res)


def num_tx_azim_antenna(cfg, mask=(True, False, True)):
    return num_tx_antenna(cfg, mask)


def num_tx_elev_antenna(cfg, mask=(False, True, False)):
    return num_tx_antenna(cfg, mask)


def num_rx_antenna(cfg):
    return sum(dec2bit(cfg['channelCfg']['rxMask'], 3))


def num_virtual_antenna(cfg):
    return num_tx_antenna(cfg) * num_rx_antenna(cfg)
 

def chirps_per_loop(cfg):
    return (cfg['frameCfg']['endIndex'] - cfg['frameCfg']['startIndex'] + 1)
    

def chirps_per_frame(cfg):
    return chirps_per_loop(cfg) * cfg['frameCfg']['loops']


def num_range_bin(cfg):
    return int(pow2_ceil(cfg['profileCfg']['adcSamples']))


def num_doppler_bin(cfg):
    return int(chirps_per_frame(cfg) / num_tx_antenna(cfg))


def num_angular_bin(cfg):
    return 64 # Hard coded

def angular_resolution(cfg):
    n = num_rx_antenna(cfg) * num_tx_azim_antenna(cfg)
    if n == 1: return float('nan')
    return math.degrees(math.asin(2 / (num_rx_antenna(cfg) * num_tx_azim_antenna(cfg))))




def range_resolution(cfg):
    return (300 * cfg['profileCfg']['sampleRate']) / (cfg['profileCfg']['adcSamples'] * 2 * cfg['profileCfg']['freqSlope'] * 1e3 )


def doppler_resolution(cfg):
    return 3e8 / (2 * cfg['profileCfg']['startFreq'] * 1e9 * (cfg['profileCfg']['idleTime'] + cfg['profileCfg']['rampEndTime']) * 1e-6 * chirps_per_frame(cfg))




