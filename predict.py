# ******************************************* #
#
# Module used to:
# - Stream and visualize Data (use mode 0 in params).
# - Stream Data and predict using traine model (use mode 1 in params).
# - Stream and save Data (use mode 2 in params).
#
# ******************************************* #


import os

import serial
import time
import datetime
import struct
import numpy as np
import scipy.interpolate as spi
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from aux.ModelClass import model_heat_vox,model_heatmapVGG,model_heatmapResNet,model_heatmap1,model_heatmap2,model_VoxelVGG,model_Voxel1 #model_knife,model_knife_spoon,model_knife_spoon_big,model_knife_spoon_ROI,model_VGGNet_32,model_CNN4_32,model_VGGNet_32_Voxel_dropout
from aux.parseTLV_full import parse_save,tlvHeaderDecode
from aux.peakGrouping import mean_range,cluster_ok,toVoxel,groupPoints

#
# Used to normalize voxels for predictions
#
class Normalize_Voxel(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std  # Note: flipping alond dim 0 will not result in a change, use with 1,2 or 3

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0},std={1})'.format(self.mean, self.std)

#
# Model loading, pick and choose (:
#
# model = torch.load("Local_Testing/Model_knife_spoon.pt", map_location=torch.device('cpu'))
# model = torch.load("Local_Testing/Model_VGGNet_32_1.pt", map_location=torch.device('cpu'))
model = torch.load("Local_Testing/trjvmltw_20.pt", map_location=torch.device('cpu'))
# model = torch.load("Local_Testing/3u6vs4fl_11.pt", map_location=torch.device('cpu'))


# Set model to eval mode
model.eval()

# To get prediction
index2class = {1:"knife", 0:"spoon"}#, 2:"hand"}#2:"nothing"}




#
# Functions to pre-process heatmap and voxels for predictions
#
def pre_process_heatmap(data):
    tensorData = torch.from_numpy(data).view([1, 1, 32, 32])
    tensorData = transforms.Normalize(mean=819.3264,std=2452.4735)(tensorData)
    # tensorData = trans(tensorData)
    return tensorData
def pre_process_voxel(data):
    tensorData = torch.from_numpy(data).view([1, 1, 26, 26, 26])
    tensorData = Normalize_Voxel(mean=4.0342, std=159.1961)(tensorData)
    # tensorData = trans(tensorData)
    return tensorData
#
# Function to emit prediction given only the heatmap
#
def pred(polarHeatmap):
    tensorHeatmap = pre_process_heatmap(polarHeatmap)
    res = model(tensorHeatmap.float())
    pred = torch.argmax(res)
    conf_value = torch.max(res)
    return index2class[pred.item()],conf_value

#
# Function to emit prediction given the heatmap and voxels.
#
def pred_heat_voxel(polarHeatmap,voxel):
    tensorHeatmap = pre_process_heatmap(polarHeatmap)
    tensorVoxel = pre_process_voxel(voxel)

    res = nn.Sigmoid()(model(tensorHeatmap.float(),tensorVoxel.float()))

    pred = res.item() > 0.5

    conf_value = res.item()
    print(conf_value)
    if pred:
        return "knife",conf_value
    else:
        return "spoon", conf_value



###################################

# Change the configuration file name
configFileName = 'Local_Testing/knife3dNoGrouping.cfg'

CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2 ** 15, dtype='uint8')
byteBufferLength = 0;


# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    # CLIport = serial.Serial('/dev/ttyACM0', 115200)
    # Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # Windows
    # CLIport = serial.Serial('COM3', 115200)
    # Dataport = serial.Serial('COM4', 921600)

    # Apparently macos (found in issues of the github): Maybe need to change(see screenshot)
    CLIport = serial.Serial("/dev//tty.usbmodemR10310411", 115200)  # 'COM3', 115200)
    Dataport = serial.Serial("/dev//tty.usbmodemR10310414", 921600)  # 'COM4', 921600)
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i + '\n').encode())
        print(i)
        time.sleep(0.01)

    return CLIport, Dataport


# ------------------------------------------------------------------

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 2  # 3 # I changed....

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1;

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;

            digOutSampleRate = int(splitWords[11]);

        # Get the information about the frame configuration
        elif "frameCfg" in splitWords[0]:

            chirpStartIdx = int(splitWords[1]);
            chirpEndIdx = int(splitWords[2]);
            numLoops = int(splitWords[3]);
            numFrames = int(splitWords[4]);
            framePeriodicity = int(splitWords[5]);

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (
                2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return configParameters


# ------------------------------------------------------------------

#
# Funtion to read and parse the incoming data, this one is used for saving data,
# the other one for predicting and visualizing.
#
def readAndParseData14xx(Dataport, configParameters, save=None):
    global byteBuffer, byteBufferLength

    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12;
    BYTE_VEC_ACC_MAX_SIZE = 2 ** 15;
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1;
    MMWDEMO_UART_MSG_RANGE_PROFILE = 2;
    maxBufferSize = 2 ** 15;
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    dataOK = 0  # Checks if the data has been read correctly
    frameNumber = 0
    detObj = []
    max_x, max_y, max_z, min_x, min_y, min_z = 0, 0, 0, 10, 10, 10
    rawHeatmap = []

    readBuffer = Dataport.read(Dataport.in_waiting)

    if save is not None:
        rawOut.write(readBuffer)

    byteVec = np.frombuffer(readBuffer, dtype='uint8')

    byteCount = len(byteVec)

    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount

    # Check that the buffer has some data
    if byteBufferLength > 16:

        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc + 8]
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:

            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength - startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength - startIdx[0]:]),
                                                                       dtype='uint8')
                byteBufferLength = byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

        # Initialize the pointer index
        idX = 0

        # Read the header
        magicNumber = byteBuffer[idX:idX + 8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4

        # UNCOMMENT IN CASE OF SDK 2
        # subFrameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        # idX += 4

        # Read the TLV messages
        print("Reading frame :     %d" %frameNumber)
        # print(numTLVs)
        for tlvIdx in range(numTLVs):

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # # Check the header of the TLV message
            # tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
            # idX += 4
            # tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
            # idX += 4
            tlv_type, tlv_length = struct.unpack('<2I', byteBuffer[idX:idX + 8])
            idX += 8

            print(tlv_type)
            # Read the data depending on the TLV message
            if tlv_type == 1:
                data = byteBuffer.tobytes()
                num_obj, xyzQFormat = struct.unpack('<2H', data[:4])
                idX+=4

                if num_obj == 0:
                    detObj = None
                else:
                    print(numDetectedObj)
                    max_x, max_y, max_z, min_x, min_y, min_z = 0, 0, 0, 10, 10, 10 # Keyword : New for obj!
                    for i in range(numDetectedObj):
                        rangeIdx, dopplerIdx, peakVal, x, y, z = struct.unpack('<HhHhhh', data[4 + 12 * i: 4 + 12 * i + 12])
                        idX += 12
                        x_coor = (x * 1.0 / (1 << xyzQFormat))
                        y_coor = (y * 1.0 / (1 << xyzQFormat))
                        z_coor = (z * 1.0 / (1 << xyzQFormat))
                        if (rangeIdx > 7) and (rangeIdx < 30) :
                            print("append")
                            detObj.append(
                                {"dopplerIdx": dopplerIdx, "rangeIdx": rangeIdx, "peakVal": peakVal,
                                 "x_coor": x_coor, "y_coor": y_coor, "z_coor": z_coor})
                            if x_coor > max_x:
                                max_x = x_coor
                            if y_coor > max_y:
                                max_y = y_coor
                            if z_coor > max_z:
                                max_z = z_coor
                            if x_coor < min_x:
                                min_x = x_coor
                            if y_coor < min_y:
                                min_y = y_coor
                            if z_coor < min_z:
                                min_z = z_coor

            if tlv_type == 4 or tlv_type > 451156100:
                data = byteBuffer.tobytes()
                for i in range(configParameters["numRangeBins"]):
                    row = []
                    for j in range(8):
                        R, I = struct.unpack('<hh', data[(4 * j) + (32 * i):(4 * (j + 1)) + (32 * i)])
                        idX += 4
                        row.append(complex(R, I))
                    rawHeatmap.append(row)
                dataOK = 1

                # Remove already processed data
        if idX > 0 and byteBufferLength > idX:
            shiftSize = totalPacketLen

            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),
                                                                 dtype='uint8')
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

    return dataOK, frameNumber, rawHeatmap, detObj, max_x,max_y,max_z,min_x,min_y,min_z



#
# Funtion to read and parse the incoming data, this one is used for predicting/visualizing live,
# the other one for saving data.
#
def readAndParseData(Dataport, configParameters, save=None):
    global byteBuffer, byteBufferLength

    maxBufferSize = 2 ** 15;
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    dataOK = 0  # Checks if the data has been read correctly
    frameNumber = 0
    detObj = []
    max_x, max_y, max_z, min_x, min_y, min_z = 0, 0, 0, 10, 10, 10
    rawHeatmap = []

    readBuffer = Dataport.read(Dataport.in_waiting)

    # If mode 2 : only read and save mode
    if save is not None:
        rawOut.write(readBuffer)


    byteVec = np.frombuffer(readBuffer, dtype='uint8')

    byteCount = len(byteVec)

    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount

    # Check that the buffer has some data
    if byteBufferLength > 16:

        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc + 8]
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:

            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength - startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength - startIdx[0]:]),
                                                                       dtype='uint8')
                byteBufferLength = byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

        # Initialize the pointer index
        idX = 8
        data = byteBuffer.tobytes()
        # skip Magic
        data = data[8:]

        version, length, platform, frameNum, cpuCycles, numObj, numTLVs = struct.unpack('<7I', data[:28])
        data = data[28:]
        idX += 28

        # Read the TLV messages
        # print("Total packet len :     %d" %totalPacketLen)
        # print("FrameNum :     %d" % frameNum)
        # print(numTLVs)
        for tlvIdx in range(numTLVs):

            # data = byteBuffer.tobytes()
            # Check the header of the TLV message
            tlv_type,tlv_length = struct.unpack('<2I', data[:8])
            data = data[8:]
            idX += 8

            # Read the data depending on the TLV message
            if tlv_type == 1:
                inital_offset = 4
                num_obj, xyzQFormat = struct.unpack('<2H', data[:inital_offset])
                if num_obj == 0:
                    detObj = None
                else:
                    max_x, max_y, max_z, min_x, min_y, min_z = 0, 0, 0, 10, 10, 10 # Keyword : New for obj!
                    for i in range(num_obj):
                        rangeIdx, dopplerIdx, peakVal, x, y, z = struct.unpack('<HhHhhh', data[inital_offset + 12 * i: inital_offset + 12 * i + 12])
                        idX += 12
                        x_coor = (x * 1.0 / (1 << xyzQFormat))
                        y_coor = (y * 1.0 / (1 << xyzQFormat))
                        z_coor = (z * 1.0 / (1 << xyzQFormat))

                        if (rangeIdx > 6) and (rangeIdx < 15) :#15
                            detObj.append(
                                {"objId":i,"dopplerIdx": dopplerIdx, "rangeIdx": rangeIdx, "peakVal": peakVal,
                                 "x_coor": x_coor, "y_coor": y_coor, "z_coor": z_coor})
                            if x_coor > max_x:
                                max_x = x_coor
                            if y_coor > max_y:
                                max_y = y_coor
                            if z_coor > max_z:
                                max_z = z_coor
                            if x_coor < min_x:
                                min_x = x_coor
                            if y_coor < min_y:
                                min_y = y_coor
                            if z_coor < min_z:
                                min_z = z_coor
            if tlv_type == 4 or tlv_type > 100:
                # data = byteBuffer.tobytes()
                for i in range(configParameters["numRangeBins"]):
                    row = []
                    for j in range(8):
                        R, I = struct.unpack('<hh', data[(4 * j) + (32 * i): (4 * (j + 1)) + (32 * i)])
                        row.append(complex(R, I))
                    rawHeatmap.append(row)
                dataOK = 1

                # Remove already processed data

            data = data[tlv_length:]
            idX += tlv_length

        if idX > 0 and byteBufferLength > idX:
            shiftSize = totalPacketLen

            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),
                                                                 dtype='uint8')
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

    return dataOK, frameNumber, rawHeatmap, detObj, max_x,max_y,max_z,min_x,min_y,min_z

# ------------------------------------------------------------------

#
# Function to process the RA heatmap values.
#
def process_RA(rawHeatmap,angle_bins=64,tx_azimuth_antennas=2,rx_antennas=4):
    if rawHeatmap == [] :
        return []
    else:
        polarHeatmap = np.reshape(rawHeatmap, (configParameters["numRangeBins"], tx_azimuth_antennas * rx_antennas))
        polarHeatmap = np.fft.fft(rawHeatmap, angle_bins)

        polarHeatmap = np.abs(polarHeatmap)
        polarHeatmap = np.fft.fftshift(polarHeatmap,axes=(1,))
        polarHeatmap = polarHeatmap[:,1:]

        return polarHeatmap

#
# Func to search for a peak in a window aroung the given range,
# Usage: If frame comes with no "detected object", we use the previous frame's peak, and search around the same coord.
#
def search_peak(processed_heatmap, rangeIdx, window=2):
    # Make sure not to search near the (0,0) coor, there is frequently antenna coupling noise there
    if rangeIdx - window <= 4:
        peak_x, peak_y = np.unravel_index(processed_heatmap[rangeIdx, :].argmax(), processed_heatmap.shape)
        peak_x += rangeIdx
    else:
        peak_x, peak_y = np.unravel_index(processed_heatmap[rangeIdx - window:rangeIdx + window:, :].argmax(),
                                          processed_heatmap.shape)
        peak_x += rangeIdx - window
    return peak_x, peak_y

#
# Func to get the ROI data given the rangeIdx detected by the on chip processing
#
def get_ROI(processed_heatmap, rangeIdx, size_x=16,size_y=16, skip_num=5):#size=16, skip_num=5):
    peak_x, peak_y = search_peak(processed_heatmap, rangeIdx)
    shape = processed_heatmap.shape

    # Initialise ROI array
    ROI = np.zeros((2 * size_x, 2 * size_y))

    # Indexes of ROI array to replace
    ROIstart_x, ROIstart_y, ROIend_x, ROIend_y = 0, 0, 2 * size_x, 2 * size_y
    # Indexes of data array to copy, the rest will be 0's so we are padding
    data_start_x, data_start_y, data_end_x, data_end_y = peak_x - size_x, peak_y - size_y, peak_x + size_x, peak_y + size_y

    if data_start_x < 0:
        # In this case we need: ROI[diff_x_1::,:] = processed_dat[:peak_x+size,peak_y-size:peak_y+size]
        ROIstart_x = size_x - peak_x
        data_start_x = 0
    elif peak_x + size_x > shape[0]:
        # here :  ROI[:diff_x_2:,:] = processed_dat[shape[0]-peak_x::,peak_y-size:peak_y+size]
        ROIend_x = size_x - peak_x + shape[0]  # 2*size + peak_x - shape[0]
        data_start_x = peak_x  # shape[0]-peak_x
        data_end_x = shape[0]

    if data_start_y < 0:
        # here :  ROI[:,start_y::] = processed_dat[peak_x-size:peak_x+size,:peak_y+size]
        ROIstart_y = size_y - peak_y
        data_start_y = 0
    elif peak_y + size_y > shape[1]:
        # here :  ROI[::,:end_y:] = processed_dat[peak_x-size:peak_x+size, shape[1]-peak_y::]
        ROIend_y = size_y - peak_y + shape[1]
        data_start_y = peak_y - size_y
        data_end_y = shape[1]

    # Finally copy the appropriate slice
    ROI[ROIstart_x:ROIend_x:, ROIstart_y:ROIend_y:] = processed_heatmap[data_start_x:data_end_x:, data_start_y:data_end_y]

    return ROI

# ------------------------------------------------------------------

#
# Func to plot the RA hetmap if desired ( data already  processed).
# Inspired from the pymmw module.
#
def plot(polarHeatmap,angle_bins=64,tx_azimuth_antennas=2,rx_antennas=4):
    ax.clear()

    zi = spi.griddata((x.ravel(), y.ravel()), polarHeatmap.ravel(), (xi, yi), method='linear')
    zi = zi[:-1, :-1]

    cm.set_data(zi[::-1, ::-1])
    return None

#
# Funtion to update the data and display in the plot
#
def update(i,mode=0,l=5):
    #global rawHeatmap

    # Read and parse the received data
    dataOk, frameNumber, rawHeatmap, detObj, max_x,max_y,max_z,min_x,min_y,min_z = readAndParseData(Dataport, configParameters)
    if dataOk and (not rawHeatmap == []):

        polarHeatmap = process_RA(rawHeatmap)



        zi = spi.griddata((x.ravel(), y.ravel()), polarHeatmap.ravel(), (xi, yi), method='linear')
        zi = zi[:-1, :-1]
        #ax.plot(zi)
        cm.set_array(zi[::-1, ::-1])
        cm.autoscale()

    return dataOk

#
# Function to animate the plot visualizations in real time.
#
def start_animation(mode,l=5):
    global ani,xi,yi,x,y,cm
    predictions = []
    write_file("preds", predictions)

    angle_bins = 64
    tx_azimuth_antennas = 2
    rx_antennas = 4
    t = np.array(range(-angle_bins // 2 + 1, angle_bins // 2)) * (2 / angle_bins)
    t = np.arcsin(t)  # t * ((1 + np.sqrt(5)) / 2)
    r = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeResolutionMeters"]

    range_depth = configParameters["numRangeBins"] * configParameters["rangeResolutionMeters"]
    range_width, grid_res = range_depth / 2, 400

    xi = np.linspace(-range_width, range_width, grid_res)
    yi = np.linspace(0, range_depth, grid_res)
    xi, yi = np.meshgrid(xi, yi)
    x = np.array([r]).T * np.sin(t)
    y = np.array([r]).T * np.cos(t)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)  # rows, cols, idx

    fig.tight_layout(pad=2)
    cm = ax.imshow(((0,) * grid_res,) * grid_res, cmap=plt.cm.jet, extent=[-range_width, +range_width, 0, range_depth],
                   alpha=0.95)
    ax.set_title('Azimuth-Range FFT Heatmap [{};{}]'.format(angle_bins, configParameters["numRangeBins"]), fontsize=10)
    ax.set_xlabel('Lateral distance along [m]')
    ax.set_ylabel('Longitudinal distance along [m]')

    ax.plot([0, 0], [0, range_depth], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, -range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, +range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.set_ylim([0, +range_depth])
    ax.set_xlim([-range_width, +range_width])

    for i in range(1, int(range_depth) + 1):
        ax.add_patch(
            pat.Arc((0, 0), width=i * 2, height=i * 2, angle=90, theta1=-90, theta2=90, color='white', linewidth=0.5,
                    linestyle=':', zorder=1))

    ani = animation.FuncAnimation(fig, update, fargs=(mode,l,), interval=33) # fargs here determines mode...
    plt.show()


#
# Function used to aggregate all necessary functions for prediction mode.
#
def predMode(Dataport,configParameters):
    while True:
        try:
            # Update the data and check if the data is okay
            dataOk, frameNumber, rawHeatmap, detObj, max_x, max_y, max_z, min_x, min_y, min_z = readAndParseData(
                Dataport, configParameters)

            if dataOk and (not rawHeatmap == []):
                # process heatmap :
                polarHeatmap = process_RA(rawHeatmap)
                # cluster the points :
                if detObj is None:
                    grouped_objects = None
                else:
                    grouped_objects = groupPoints(detObj)

                    # print(len(grouped_objects))
                # If this frame contains enough no detected points (or not enough..),
                # consider no object detected and skip
                if grouped_objects is None or not cluster_ok(grouped_objects, max_x, max_y, max_z, min_x, min_y, min_z):
                    print(" Nothing      detected")
                    print("_________________")
                else:
                    # print(len(grouped_objects))
                    detected_point_range = mean_range(grouped_objects)
                    ROI = get_ROI(polarHeatmap, detected_point_range)
                    voxel = toVoxel(grouped_objects, 25, max_x, max_y, max_z, min_x, min_y, min_z)
                    obj, conf = pred_heat_voxel(ROI, voxel)
                    print(" %s      detected" % obj)
                    print("_________________")
            # time.sleep(1)

        # Stop the program and close everything if Ctrl + c is pressed
        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()

            break


#
# Main, will prompt for mode type, then if required, name of file to save and number of frames.
#
if __name__ == "__main__":
    # Prompt what mode to be used?


    mode = input("Mode?: ('h' for help)")
    while mode not in ["0","1","2"]:
        if mode == 'h' :
            print("0: stream and view data")
            print("1: show heatmap and prediction")
            print("2: read and save current data stream")
            mode = input("Mode?: ('h' for help)")
        else:
            mode = input("input not recognised...Mode?: ('h' for help) ")

    # Mode 0 or mode 1 is just the viewing
    l=5
    if mode == "0" :
        CLIport, Dataport = serialConfig(configFileName)
        configParameters = parseConfigFile(configFileName)
        try:
            start_animation(int(mode),int(l))

        # Stop the program and close everything if Ctrl + c is pressed
        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()

     # Mode 1 is view + pred
    if mode == "1" :
        # Start sensor
        CLIport, Dataport = serialConfig(configFileName)
        configParameters = parseConfigFile(configFileName)
        predMode(Dataport,configParameters)

    # else, Mode 2 is record
    else:
        data_name = input("name for current data stream?:")
        while len(data_name) < 3:
            data_name = input("name for current data stream?: (longer than 3 chars)")
        max_frames = int(input("number of frames to save?:"))
        while max_frames < 0:
            max_frames = int(input("number of frames to save?: (need positive integer)"))

        # Create directory to save data if it doesnt exist..
        dataPath = os.path.join(os.getcwd(), "Data/knife_spoon_test_set")
        if not os.path.exists(dataPath):
            os.mkdir(dataPath)
        path = os.path.join(dataPath, data_name)
        if not os.path.exists(path):
            os.mkdir(path)

        # Read and save input until prompted to stop, or until number of frames reached:
        full_name = path + "/" + data_name + ".dat"
        rawOut = open(full_name, "wb")
        frameNumber = 0

        # Start sensor
        CLIport, Dataport = serialConfig(configFileName)
        configParameters = parseConfigFile(configFileName)

        # while frameNumber not reached:
        while frameNumber <= max_frames:
            _, frameNumber, _,_,_,_,_,_,_,_ = readAndParseData14xx(Dataport, configParameters, save=rawOut)

        rawOut.close()
        # Stop sensor
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()

        # Now call auxiliary parsing function
        parse_save(path,data_name, int(configParameters["numRangeBins"]), int(configParameters["numDopplerBins"]),
                   configParameters["rangeResolutionMeters"], tx_ant=3, rx_ant=4)




























