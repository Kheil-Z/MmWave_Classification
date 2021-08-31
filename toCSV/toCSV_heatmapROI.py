# ******************************************* #
#
# Module used to transfrom parsed .txt data to .csv used in learning notebook.
# This module only incorporates the ROI from the Heatmap which I tried using at first,
# if you wish to use it you must change the predict function to be compatible with such data input.
#
# ******************************************* #



import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import scipy.interpolate as spi
import matplotlib.patches as pat
import csv

from os import listdir
from parseCFG import *
from peakGrouping import *

#
# Function to read Range-Azimuth heatmap structure from parsed txt file
#
def read_RA(name):
    f = open(name, "r")
    while (f.readline().rstrip() != "TLV- Azimuth Static Heatmap:"):
        continue
    res = []
    while True:
        x = f.readline()
        if x.rstrip()[1:5] != "Prof":
            break
        else:
            d = []
            for s in re.findall(r'.\d+.\d+\w', x)[1::]:
                d.append(complex(s))
            res.append(d)
    f.close()
    return res

#
# Function to read detected points structure from parsed txt file
#
def read_detected_points(name):
    f = open(name, "r")
    while "Frame" not in f.readline().rstrip():
        continue
    curr_line = f.readline()
    num_obj = int(curr_line[12::])
    if num_obj == 0:
        return None
    else:
        res = []
        while True:
            x = f.readline()
            if "TLV- Azimuth Static Heatmap:" in x:
                break
            if "DopplerIdx" in x:
                dopplerIdx = int(x[13::])
                x = f.readline()
                rangeIdx = int(x[11::])

                x = f.readline()
                peakVal = int(x[10::])

                x = f.readline()
                x_coor = float(x[4::])
                x = f.readline()
                y_coor = float(x[4::])
                x = f.readline()
                z_coor = float(x[4::])

                # In this structure (only heatmap, all we need is the rangeIdx t get the ROI)
                if (rangeIdx > 7) and (rangeIdx < 60) and (rangeIdx not in res) and (peakVal > 200):  # y_coor > 0.2 and y_coor < 1 :
                    # res.append(dopplerIdx,rangeIdx,x_coor,y_coor,z_coor)
                    res.append(rangeIdx)
        f.close()
        return res

#
# Function to process the RA heatmap values.
#
def process_RA(rawDat, range_bins, angle_bins, tx_azimuth_antennas=2, rx_antennas=4):
    a = np.reshape(rawDat, (range_bins, tx_azimuth_antennas * rx_antennas))
    a = np.fft.fft(rawDat, angle_bins)

    a = np.abs(a)
    a = np.fft.fftshift(a, axes=(1,))
    a = a[:, 1:]

    return a

#
# Func to search for a peak in a window aroung the given range,
# Usage: If frame comes with no "detected object", we use the previous frame's peak, and search around the same coord.
#
def search_peak(processed_heatmap, rangeIdx, window=2):
    # Make sure not to search near the (0,0) coor, there is frequently antenna coupling noise there
    if rangeIdx - window < 5:
        peak_x, peak_y = np.unravel_index(processed_heatmap[rangeIdx, :].argmax(), processed_heatmap.shape)
        peak_x += rangeIdx
    else:
        peak_x, peak_y = np.unravel_index(processed_heatmap[rangeIdx - window:rangeIdx + window:, :].argmax(),
                                          processed_heatmap.shape)
        peak_x += rangeIdx - window
    return peak_x, peak_y


#
# # Func to get the ROI data given the rangeIdx detected by the on chip processing
# def get_ROI(processed_heatmap, rangeIdx, size_x=16,size_y=16, skip_num=5):#size=16, skip_num=5):
#     peak_x, peak_y = search_peak(processed_heatmap, rangeIdx)
#     shape = processed_heatmap.shape
#
#     # Initialise ROI array
#     ROI = np.zeros((2 * size_x, 2 * size_y))
#
#     # Indexes of ROI array to replace
#     ROIstart_x, ROIstart_y, ROIend_x, ROIend_y = 0, 0, 2 * size_x, 2 * size_y
#     # Indexes of data array to copy, the rest will be 0's so we are padding
#     data_start_x, data_start_y, data_end_x, data_end_y = peak_x - size_x, peak_y - size_y, peak_x + size_x, peak_y + size_y
#
#     if data_start_x < 0:
#         # In this case we need: ROI[diff_x_1::,:] = processed_dat[:peak_x+size,peak_y-size:peak_y+size]
#         ROIstart_x = size_x - peak_x
#         data_start_x = 0
#     elif peak_x + size_x > shape[0]:
#         # here :  ROI[:diff_x_2:,:] = processed_dat[shape[0]-peak_x::,peak_y-size:peak_y+size]
#         ROIend_x = size_x - peak_x + shape[0]  # 2*size + peak_x - shape[0]
#         data_start_x = peak_x  # shape[0]-peak_x
#         data_end_x = shape[0]
#
#     if data_start_y < 0:
#         # here :  ROI[:,start_y::] = processed_dat[peak_x-size:peak_x+size,:peak_y+size]
#         ROIstart_y = size_y - peak_y
#         data_start_y = 0
#     elif peak_y + size_y > shape[1]:
#         # here :  ROI[::,:end_y:] = processed_dat[peak_x-size:peak_x+size, shape[1]-peak_y::]
#         ROIend_y = size_y - peak_y + shape[1]
#         data_start_y = peak_y - size_y
#         data_end_y = shape[1]
#
#     # Finally copy the appropriate slice
#     ROI[ROIstart_x:ROIend_x:, ROIstart_y:ROIend_y:] = processed_heatmap[data_start_x:data_end_x:, data_start_y:data_end_y]
#
#     return ROI

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
        data_start_x = peak_x -size_x # shape[0]-peak_x
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


#
# Func to get the label of the data, if not using my data, should rewrite.
#
def get_label(name):
    if "knife" in name:
        return "knife"
    elif "spoon" in name:
        return "spoon"
    else:
        return "nothing"


#
# Func to plot the RA hetmap if desired (transforms to cartesian coords for better visuals).
# Inspired from the pymmw module.
#
def plot_RA(processed_RA, range_bins, angle_bins, range_res, tx_azimuth_antennas=2, rx_antennas=4):
    t = np.array(range(-angle_bins // 2 + 1, angle_bins // 2)) * (2 / angle_bins)
    t = np.arcsin(t)  # t * ((1 + np.sqrt(5)) / 2)
    r = np.array(range(range_bins)) * range_res

    range_depth = range_bins * range_res
    range_width, grid_res = range_depth / 2, 400

    xi = np.linspace(-range_width, range_width, grid_res)
    yi = np.linspace(0, range_depth, grid_res)
    xi, yi = np.meshgrid(xi, yi)
    x = np.array([r]).T * np.sin(t)
    y = np.array([r]).T * np.cos(t)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)  # rows, cols, idx

    fig.tight_layout(pad=2)
    cm = ax.imshow(((0,) * grid_res,) * grid_res, cmap=plt.cm.jet, extent=[-range_width, +range_width, 0, range_depth],
                   alpha=0.95)
    ax.set_title('Azimuth-Range FFT Heatmap [{};{}]'.format(angle_bins, range_bins), fontsize=10)
    ax.set_xlabel('Lateral distance along [m]')
    ax.set_ylabel('Longitudinal distance along [m]')

    ax.plot([0, 0], [0, range_depth], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, -range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, +range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.set_ylim([0, +range_depth])
    ax.set_xlim([-range_width, +range_width])

    zi = spi.griddata((x.ravel(), y.ravel()), processed_RA.ravel(), (xi, yi), method='linear')
    zi = zi[:-1, :-1]
    cm.set_array(zi[::-1,::-1])  # rotate 180 degrees
    for i in range(1, int(range_depth)+1):
        ax.add_patch(pat.Arc((0, 0), width=i*2, height=i*2, angle=90, theta1=-90, theta2=90, color='white', linewidth=0.5, linestyle=':', zorder=1))
    cm.autoscale()
    plt.show()


#
# Func used to order frames by number when processing.
#
def key(s):

    if s[-7] in '0123456789':
        return int(s[-7:-4])
    if s[-6] in '0123456789':
        return int(s[-6:-4])
    else:
         return int(s[-5:-4])


#
# Main, requires input folder of data and targetforlder to save .csv files
#
if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: toCSV_heatmapROI.py <inputfolder>  <targetFolder>")
        sys.exit()

    root_folder = sys.argv[1]
    sub_folders = listdir(root_folder)
    files = []
    target_folder = sys.argv[2]

    # read necessary configurations from cfg file
    f_cfg = open("test_crop/LbConfig.cfg", "r")
    cfg = parseCFG(f_cfg, 2, 4)
    f_cfg.close()
    range_bin, ang_bin = num_range_bin(cfg), num_angular_bin(cfg)
    range_res, ang_res = range_resolution(cfg), angular_resolution(cfg)

    with open(target_folder + "mmwaveRA.csv", mode="w", newline='') as RA:
        RA_writer = csv.writer(RA, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['filename', ' Label']
        RA_writer.writerow(header)

        print(sub_folders)

        for folder in sub_folders:
            print(folder)
            print("################################")
            if (not ".DS" in folder) and (not ".dat" in folder):
                curr_path = root_folder + "/" + folder + "/txt"
                txts = listdir(curr_path)
                for name in txts:
                    curr_name = curr_path + "/" + name
                    if (not ".DS" in curr_name):
                        print("________")
                        print(curr_name)
                        heatmap_id = target_folder + "/" + folder + "_" + name[:-4:]
                        # Read the raw .txt data to extract RA heatmap and detected points
                        dat = read_RA(curr_name)
                        new_detected_points = read_detected_points(curr_name)
                        # If this frame contains no detected points, use previous one's point(s)
                        if new_detected_points is not None:
                            detected_points = new_detected_points
                        # extract the RA heatmap (in polar coordinates )
                        processed_heatmap = process_RA(dat, range_bin, ang_bin)

                        if "behind" in curr_name:
                            # get the labels
                            labels = get_label_several(curr_name)
                            i = 0
                            if detected_points is []:
                                print("#########______EMPTYYYYYYYYY______##########")
                            else:
                                detected_points.sort()
                                for rangeIdx in detected_points[::2]:
                                    ROI = get_ROI(processed_heatmap, rangeIdx, size_x=16,size_y=16)  # size=16) #16 - gives 32x32
                                    # Save the heatmap ROI, and add it to the dataset csv
                                    RA_writer.writerow([folder + "_" + name[:-4:] + "_" + str(i) + ".csv", labels[i]])

                                    np.savetxt(heatmap_id + "_" + str(i) + ".csv", ROI, delimiter=",")

                                    i += 1
                        else:
                            # get the label
                            label = get_label(curr_name)
                            i = 0

                            if detected_points is []:
                                print("#########______EMPTYYYYYYYYY______##########")
                            else:
                                for rangeIdx in detected_points:
                                    ROI = get_ROI(processed_heatmap, rangeIdx, size_x=16,size_y=16)  # size=16) #16 - gives 32x32
                                    print(ROI.shape)
                                    # Save the heatmap ROI, and add it to the dataset csv
                                    # RA_writer.writerow([heatmap_id+"_"+str(i), label])
                                    RA_writer.writerow([folder + "_" + name[:-4:] + "_" + str(i) + ".csv", label])

                                    np.savetxt(heatmap_id + "_" + str(i) + ".csv", ROI, delimiter=",")

                                    # plt.imsave("test_crop/FPS/" + folder +name[:-4:]+"_"+str(i) + ".png", ROI)
                                    i += 1