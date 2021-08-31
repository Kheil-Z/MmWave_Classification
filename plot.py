# ******************************************* #
#
# Module used to plot data if you wish to visualize after saving.
# Heavily inspired by the pymmw Mmwave module (https://github.com/m6c7l/pymmw)
#
# ******************************************* #

import numpy as np
import matplotlib.pyplot as plt
import re
import sys

import scipy.interpolate as spi
import matplotlib.patches as pat

from os import listdir
from parseCFG import *
from get_mask import *

#
# Function to read Range-Azimuth heatmap structure from parsed txt file
#
def read_RA(name):
    f = open(name,"r")
    while (f.readline().rstrip() != "TLV- Azimuth Static Heatmap:"):
        continue
    res = []
    while True :
        x = f.readline()
        if x.rstrip()[1:5] != "Prof":
            break
        else :
            d = []
            for s in re.findall(r'.\d+.\d+\w',x)[1::]:
                d.append(complex(s))
            res.append(d)
    f.close()
    return res

#
# Function to read Range-Doppler heatmap structure from parsed txt file
#
def read_RD(name):
    f = open(name,"r")
    while (f.readline().rstrip() != "TLV-Range/Doppler Heatmap:"):
        continue
    res = []
    while True :
        x = f.readline()
        if x.rstrip()[1:5] != "Prof":
            break
        else :
            d = [int(i)/512 for i in x.rstrip()[15::].split()]
            res.append(d)
    f.close()
    return res

#
# Func to plot the RD hetmap if desired.
#
def process_RD(dat,range_bins, doppler_bins, res_range,res_doppler ,tx_azimuth_antennas=2,rx_antennas=4):
 # ---        
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)  # rows, cols, idx
    ax.set_title('Doppler-Range FFT Heatmap [{};{}]'.format(range_bins, doppler_bins), fontsize=10)
    ax.set_xlabel('Longitudinal distance [m]')
    ax.set_ylabel('Radial velocity [m/s]')
    ax.grid(color='white', linestyle=':', linewidth=0.5)
    
    scale = max(doppler_bins, range_bins)
    ratio = range_bins / doppler_bins
    range_offset = res_range / 2
    range_min = 0 - range_offset
    range_max = range_min + scale * res_range
    doppler_scale = scale // 2 * res_doppler
    doppler_offset = 0 # (res_doppler * ratio) / 2
    doppler_min = (-doppler_scale + doppler_offset) / ratio
    doppler_max = (+doppler_scale + doppler_offset) / ratio
    
    im = ax.imshow(np.reshape([0,] * range_bins * (doppler_bins-1), (range_bins, doppler_bins-1)),
                       cmap=plt.cm.jet,
                       interpolation='quadric',  # none, gaussian, mitchell, catrom, quadric, kaiser, hamming
                       aspect=(res_range / res_doppler) * ratio,
                       extent=[range_min, range_max, doppler_min, doppler_max], alpha=.95)
    ax.plot([0, range_max], [0, 0], color='white', linestyle=':', linewidth=0.5, zorder=1)


    b = np.reshape(dat, (range_bins, doppler_bins))
    c = np.fft.fftshift(b, axes=(1,))  # put left to center, put center to right
   # print(c[:,1:].T)    
    im.set_array(c[:,1:].T)
    im.autoscale()
   # plt.show()


#
# Func to process and plot the RA hetmap if desired (transforms to cartesian coords for better visuals).
#
def process_RA(name, dat,range_bins, angle_bins, range_res ,tx_azimuth_antennas=2,rx_antennas=4):
    
    t = np.array(range(-angle_bins//2 + 1, angle_bins//2)) * (2 / angle_bins)
    t = np.arcsin(t) # t * ((1 + np.sqrt(5)) / 2)
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
    cm = ax.imshow(((0,)*grid_res,) * grid_res, cmap=plt.cm.jet, extent=[-range_width, +range_width, 0, range_depth], alpha=0.95)
    ax.set_title('Azimuth-Range FFT Heatmap [{};{}]'.format(angle_bins, range_bins), fontsize=10)
    ax.set_xlabel('Lateral distance along [m]')
    ax.set_ylabel('Longitudinal distance along [m]')

    ax.plot([0, 0], [0, range_depth], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, -range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.plot([0, +range_width], [0, range_width], color='white', linewidth=0.5, linestyle=':', zorder=1)
    ax.set_ylim([0, +range_depth])
    ax.set_xlim([-range_width, +range_width])
    
    
    a = np.reshape(dat, (range_bins, tx_azimuth_antennas * rx_antennas))
    a = np.fft.fft(dat, angle_bins)
     
    a = np.abs(a)
    a = np.fft.fftshift(a, axes=(1,))  # put left to center, put center to right       
    a = a[:,1:]  # cut off first angle bin
      
    zi = spi.griddata((x.ravel(), y.ravel()), a.ravel(), (xi, yi), method='linear')
    zi = zi[:-1,:-1]

    
    cm.set_array(zi[::-1,::-1])  # rotate 180 degrees
    for i in range(1, int(range_depth)+1):
        ax.add_patch(pat.Arc((0, 0), width=i*2, height=i*2, angle=90, theta1=-90, theta2=90, color='white', linewidth=0.5, linestyle=':', zorder=1))
    cm.autoscale()
    #plt.show()
    return a

#
# Key to sort frames correctly
#
def key(s):
    if s[-7] in '0123456789':
        return int(s[-7:-4])
    if s[-6] in '0123456789':
        return int(s[-6:-4])
    else:
         return int(s[-5:-4])

if __name__ == "__main__":
    path = sys.argv[1]
    type = sys.argv[2]

    if len(sys.argv) == 4 :
        tx_ant, rx_ant = sys.argv[3], sys.argv[4]
    else :
        tx_ant, rx_ant = 2, 4
        print("_________")
        print("     Note : TX and RX not specified, used 4, 2")
        print("_________")
    
    f_cfg = open("test_crop/LbConfig.cfg","r")
    cfg = parseCFG(f_cfg, tx_ant, rx_ant)
    f_cfg.close()

    res_range, res_doppler = range_resolution(cfg), doppler_resolution(cfg)
    range_bin, doppler_bin, ang_bin = num_range_bin(cfg), num_doppler_bin(cfg), num_angular_bin(cfg)
    print(res_range)

    if type == "1" :
        for n in files:
            dat = process(n)
            plot(dat)
    elif type =="2":
        for n in files:
            dat = process(n)
            plot2(dat)
    elif type =="3":
        for fileName in listdir(path):
            n = path+"/"+fileName
            if not ".DS" in n :
                print(fileName)
                dat = read_RD(n)

                process_RD(dat,range_bin, doppler_bin, res_range,res_doppler ,tx_azimuth_antennas=tx_ant,rx_antennas=rx_ant)
        plt.show()
    else:
        print(listdir(path))
        files = listdir(path)
        files.sort(key=key)
        for fileName in files:
            n = path+"/"+fileName
            if not ".DS" in n :
#                print(fileName)
                dat = read_RA(n)
                zi = process_RA(fileName,dat,range_bin,ang_bin,res_range)
                plt.show()
                plt.imshow(zi)
                # plt.imsave("report_ex.jpg",zi)
                plt.show()
                ROI = get_ROI(zi,size=16)
                plt.imshow(ROI)
                # plt.imsave("report_ex.jpg",zi)
                plt.show()
                # plt.imshow(zi)
                # plt.show()
