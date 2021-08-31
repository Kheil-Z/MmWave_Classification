import os
import struct
import sys
import math
from pathlib import Path
import re

from parseCFG import *

# CLI Profile Configuration Variables
startFreq = 77  # GHz
# chirpMargin_default = 0  #Might change with frame... gonna ignore for now
numChirpsPerFrame = 64
frameDuration = 100000  # usec or 100 msec
# chirpingTime        = frameDuration - chirpMargin_default
speedOfLight = 299792458.0  # m/s

magic = b'\x02\x01\x04\x03\x06\x05\x08\x07'

#### Parsing Functions ###

def tlvHeaderDecode(data):
    tlvType, tlvLength = struct.unpack('<2I', data)
    return tlvType, tlvLength


def parseDetectedObjects(data):
    inital_offset = 4
    numDetectedObj, xyzQFormat = struct.unpack('<2H', data[:inital_offset])
    # Write
    out.write("TLV-Objects:\n")
    out.write("Detect Obj:%d\n" % (numDetectedObj))

    for i in range(numDetectedObj):
        rangeIdx, dopplerIdx, peakVal, x, y, z = struct.unpack('<HhHhhh', data[
                                                                          inital_offset + 12 * i: inital_offset + 12 * i + 12])
        # Write
        out.write("\tObjId:%d \n" % (i))
        out.write("\t\tDopplerIdx:%d \n" % (dopplerIdx))
        out.write("\t\tRangeIdx:%d \n" % (rangeIdx))
        out.write("\t\tPeakVal:%d \n" % (peakVal))
        out.write("\t\tX:%07.3f \n" % (x * 1.0 / (1 << xyzQFormat)))
        out.write("\t\tY:%07.3f \n" % (y * 1.0 / (1 << xyzQFormat)))
        out.write("\t\tZ:%07.3f \n" % (z * 1.0 / (1 << xyzQFormat)))
        out.write("\t\tRange:%07.3fm \n" % (math.sqrt(
            pow((x * 1.0 / (1 << xyzQFormat)), 2) + pow((y * 1.0 / (1 << xyzQFormat)), 2) + pow(
                (z * 1.0 / (1 << xyzQFormat)), 2))))

        out.write(
            "\t\tVelocity:%0.3fm/s \n" % (dopplerIdx * (speedOfLight / (2 * (startFreq * 1e9) * frameDuration * 1e-6))))


def parseRangeProfile(data, tlvLength):
    out.write("TLV-Range Profile: \n")
    for i in range(numRangeBins):
        rangeProfile = struct.unpack('<H', data[2 * i:2 * i + 2])
        out.write("\tRangeProf[%0.3fm]: %07.3fdB \n" % (
        i * rangeResolution, 20 * math.log10(2 ** (rangeProfile[0] / (2 ** 9)))))


def parseNoiseProfile(data):
    out.write("TLV-Noise Profile: \n")
    for i in range(numRangeBins):
        noiseProfile = struct.unpack('<H', data[2 * i:2 * i + 2])
        out.write("\tNoiseFloorProf[%0.3fm]: %07.3f \n" % (
        i * rangeResolution, 20 * math.log10(2 ** (noiseProfile[0] / (2 ** 9)))))


def azimuthStaticHeatmap(data):
    out.write("TLV- Azimuth Static Heatmap: \n")
    for i in range(numRangeBins):
        out.write("\tProf[%0.3fm]: " % (i * rangeResolution))
        for j in range(8):
            R, I = struct.unpack('<hh', data[(4 * j) + (32 * i):(4 * (j + 1)) + (32 * i)])
            if I >= 0:
                out.write("%d+%dj " % (R, I))
            else:
                out.write("%d-%dj " % (R, -I))
        out.write("\n")


def rangeDopplerHeatmap(data):
    out.write("TLV-Range/Doppler Heatmap:\n")
    for i in range(numRangeBins):
        out.write("\tProf[%0.3fm]: " % (i * rangeResolution))
        for j in range(numDopplerBins):
            X = struct.unpack('<h', data[(2 * j) + (numDopplerBins * i):(2 * (j + 1)) + (numDopplerBins * i)])
            out.write("%d " % (X))
        out.write("\n")


def parseStats(data, tlvLength):
    interProcess, transmitOut, frameMargin, chirpMargin, activeCPULoad, interCPULoad = struct.unpack('<6I',
                                                                                                     data[:tlvLength])
    # Write
    out.write("TLV-Stats:")
    out.write("\tOutputMsgStats:\t%d " % (6))
    out.write("\t\tChirpMargin:\t%d " % (chirpMargin))
    out.write("\t\tFrameMargin:\t%d " % (frameMargin))
    out.write("\t\tInterCPULoad:\t%d " % (interCPULoad))
    out.write("\t\tActiveCPULoad:\t%d " % (activeCPULoad))
    out.write("\t\tTransmitOut:\t%d " % (transmitOut))
    out.write("\t\tInterprocess:\t%d " % (interProcess))


def tlvHeader(data, out):
    pendingBytes = 29
    while pendingBytes > 28:
        # find start magic
        offset = data.find(magic)
        data = data[offset:]
        headerLength = 28
        # Shift data off of Magic
        data = data[8:]
        try:
            version, length, platform, frameNum, cpuCycles, numObj, numTLVs = struct.unpack('<7I', data[:headerLength])
        except struct.error as e:
            print("Improper TLV structure found: ", (data,))
            print("Error ", e)
            print(pendingBytes)
            break
        out.write("Packet ID:\t%d \n" % (frameNum))
        out.write("Version:\t%x \n" % (version))
        out.write("num TLVs:\t\t%d \n" % (numTLVs))
        out.write("Frame:\t\t%d \n" % (frameNum))
        out.write("Detect Obj:\t%d \n" % (numObj))
        out.write("Platform:" + hex(platform) + "\n")

        pendingBytes = length - headerLength
        data = data[headerLength:]
        for i in range(numTLVs):
            tlvType, tlvLength = tlvHeaderDecode(data[:8])
            data = data[8:]
            if tlvType == 1:
                parseDetectedObjects(data)
            elif tlvType == 2:
                parseRangeProfile(data, tlvLength)
            elif tlvType == 3:
                parseNoiseProfile(data)
            elif tlvType == 4:
                azimuthStaticHeatmap(data)
            elif tlvType == 5:
                rangeDopplerHeatmap(data)
            elif tlvType == 6:
                parseStats(data, tlvLength)
            else:
                print("Unidentified tlv type %d" % tlvType)
            data = data[tlvLength:]
            pendingBytes -= (8 + tlvLength)


def parse_save(path, fileName, rangeBins, dopplerBins, rangeRes, tx_ant=2, rx_ant=4):
    global numRangeBins, rangeResolution, numDopplerBins, out
    numRangeBins, numDopplerBins, rangeResolution = rangeBins, dopplerBins, rangeRes

    full_name = path + "/" + fileName + ".dat"
    rawDataFile = open(full_name, "rb")
    rawData = rawDataFile.read()
    rawDataFile.close()


    # to parse each frame: Find occurences of magic
    idxs = [m.start() for m in re.finditer(magic, rawData)]
    path_txt = os.path.join(path, "txt")
    os.mkdir(path_txt)
    for i in range(len(idxs) - 1):
        # Parse for each :
        print("Parsinging frame :     %d" % i)
        name = "Frame_" + str(i) + ".txt";
        out = open(os.path.join(path_txt, name), "w+")

        tlvHeader(rawData[idxs[i]:idxs[i + 1]:], out)
        out.close()
    print("End...")
#
# if __name__ == "__main__":
#     if (len(sys.argv) != 2 ) and (len(sys.argv) != 4 ):
#         print("Usage: parseTLV.py <inputFile.bin> (optional: <tx_ant> <rx_ant>)")
#         sys.exit()
#     fileName = sys.argv[1]
#     rawDataFile = open(fileName, "rb")
#     rawData = rawDataFile.read()
#     rawDataFile.close()
#
#     if len(sys.argv) == 4 :
#         tx_ant, rx_ant = sys.argv[2], sys.argv[3]
#     else :
#         tx_ant, rx_ant = 2, 4
#         print("________")
#         print("     Note: num tx and rx not specified, so used 4, 2 .... ")
#         print("________")
#
#     f_cfg = open("profile.cfg","r")
#     cfg = parseCFG(f_cfg,tx_ant,rx_ant)
#     numRangeBins, numDopplerBins, rangeResolution = num_range_bin(cfg), num_doppler_bin(cfg), range_resolution(cfg)
#     f_cfg.close()
#
#     # Parse each frame: Find occurences of magic
#     idxs = [m.start() for m in re.finditer(magic,rawData)]
#     path = os.path.join(os.getcwd(),fileName[:-4])
#     os.mkdir(path)
#     for i in range(len(idxs)-1):
#         # Parse for each :
#         name = "Frame_"+str(i)+".txt";
#         out= open(os.path.join(path,name),"w+")
#
#         tlvHeader(rawData[idxs[i]:idxs[i+1]:])
#         #, skip_stats=True, skip_range=False)
#     print("End...")
