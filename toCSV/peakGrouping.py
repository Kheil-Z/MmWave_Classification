# ******************************************* #
#
# Module used to Voxelize point cloud, this is done essentially by applying peakGrouping operations,
# normalizing, then voxelizing. Also some interpolation functions are present.
#
# ******************************************* #


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import scipy.interpolate as spi
import matplotlib.patches as pat
import csv
from scipy.spatial import ConvexHull
from os import listdir
# from parseCFG import *

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
    max_x,max_y,max_z,min_x,min_y,min_z = 0,0,0,10,10,10
    f = open(name, "r")
    while "Frame" not in f.readline().rstrip():
        continue
    curr_line = f.readline()
    num_obj = int(curr_line[12::])
    if num_obj < 2:
        return None,0,0,0,0,0,0
    else:
        res = []
        while True:
            x = f.readline()
            if "TLV- Azimuth Static Heatmap:" in x:
                break
            if "ObjId:" in x:
                objId = int(x[7::])

                x = f.readline()
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

                if (rangeIdx > 7) and (rangeIdx < 30):  # y_coor > 0.2 and y_coor < 1 :
                    # res.append(dopplerIdx,rangeIdx,x_coor,y_coor,z_coor)
                    res.append({"objId": objId, "dopplerIdx": dopplerIdx, "rangeIdx": rangeIdx, "peakVal": peakVal,
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
        f.close()
        return res,max_x,max_y,max_z,min_x,min_y,min_z

#
# Helper function to determine if integer points are neighbouring
#
def isClose(a,b):
    return np.abs(a-b) < 1.1


#
# Returns list of Object Ids grouped by range Idx
#
def rangeIdxGrouping(detectedPoints):
    detectedPoints.sort(key=lambda el: el["rangeIdx"])
    if len(detectedPoints) < 1 :
        return detectedPoints
    else:
        groups = []
        group = [detectedPoints[0]["objId"]]
        for i in range(len(detectedPoints)-1):
            if isClose(detectedPoints[i]["rangeIdx"],detectedPoints[i+1]["rangeIdx"]):
                group.append(detectedPoints[i+1]["objId"])
            else:
                groups.append(group)
                group = [detectedPoints[i+1]["objId"]]
        groups.append(group)
        return groups

#
# Returns list of Object Ids grouped by doppler Idx
#
def dopplerIdxGrouping(detectedPoints):
    detectedPoints.sort(key=lambda el: el["dopplerIdx"])
    groups = []
    group = [detectedPoints[0]["objId"]]
    for i in range(len(detectedPoints)-1):
        if isClose(detectedPoints[i]["dopplerIdx"],detectedPoints[i+1]["dopplerIdx"]):
            group.append(detectedPoints[i+1]["objId"])
        else:
            groups.append(group)
            group = [detectedPoints[i+1]["objId"]]
    groups.append(group)
    return groups

#
# Groups points by range the doppler idx
# Input : un-ordered list of points represented by dictionaries
# Output : list of grouped objects
#
def groupPoints(detectedPoints):
    res = []
    groupedByRange = []
    groupsRange = rangeIdxGrouping(detectedPoints)
    for group in groupsRange:
        groupedByRange.append(list(filter(lambda object: object['objId'] in group, detectedPoints)))

    for grouped in groupedByRange:
        groupsDoppler = dopplerIdxGrouping(grouped)
        for group in groupsDoppler:
            res.append(list(filter(lambda object: object['objId'] in group, detectedPoints)))

    # Return only the largest group:
    best_idx = 0
    if len(res) > 1:
        max_len = len(res[0])
        for i in range(len(res)):
            if len(res[i]) > max_len:
                max_len = len(res[i])
                best_idx = i
    if res == []:
        return None
    else:
        return res[best_idx]

#
# Function to normalize 3d point cloud to  [-1,1]^3:
#
def scale(pointCloud,max_x,max_y,max_z,min_x,min_y,min_z):
    scaled_cloud = []
    for dict in pointCloud:
        peakVal = dict["peakVal"]
        x_coor = (dict["x_coor"] - min_x)/(max_x - min_x)
        y_coor = (dict["x_coor"] - min_y) / (max_y - min_y)
        z_coor = (dict["x_coor"] - min_z) / (max_z - min_z)
        scaled_cloud.append({"peakVal": peakVal,"x_coor": x_coor, "y_coor": y_coor, "z_coor": z_coor})

#
# Function to voxelise a 3d point cloud where points have previously been scaled to [-1,1]
#
def toVoxel(points,voxelSize,max_x,max_y,max_z,min_x,min_y,min_z):
    voxels = np.zeros((voxelSize+1,voxelSize+1,voxelSize+1))

    seen = []
    for point in points: #zip(x_cords, y_cords, z_cords):
        x_i = int(voxelSize*(point["x_coor"] - min_x) / (max_x - min_x))
        y_i = int(voxelSize*(point["y_coor"] - min_y) / (max_y - min_y))
        z_i = int(voxelSize*(point["z_coor"] - min_z) / (max_z - min_z))
        if [x_i,y_i,z_i] not in seen:
            # First add the intensity
            voxels[x_i][y_i][z_i] += point["peakVal"]
            seen.append([x_i,y_i,z_i])

            # Then interpoalte additional points
            if x_i < voxelSize:
                voxels[x_i+1][y_i][z_i] -= np.sqrt(point["peakVal"])
            if x_i > 0:
                voxels[x_i-1][y_i][z_i] += np.sqrt(point["peakVal"])

            if y_i < voxelSize:
                voxels[x_i][y_i+1][z_i] += np.sqrt(point["peakVal"])
            if y_i > voxelSize:
                voxels[x_i][y_i-1][z_i] += np.sqrt(point["peakVal"])

            if z_i < voxelSize:
                voxels[x_i][y_i][z_i+1] += np.sqrt(point["peakVal"])
            if z_i > voxelSize:
                voxels[x_i][y_i][z_i-1] += np.sqrt(point["peakVal"])

        # # Uncomment, if you want further interpolation
        # if x_i < voxelSize:
        #     voxels[x_i+1][y_i][z_i] += point["peakVal"]/2
        # if x_i > 0:
        #     voxels[x_i-1][y_i][z_i] += point["peakVal"]/2
        #
        # if y_i < voxelSize:
        #     voxels[x_i][y_i+1][z_i] += point["peakVal"]/2
        # if y_i > voxelSize:
        #     voxels[x_i][y_i-1][z_i] += point["peakVal"]/2
        #
        # if z_i < voxelSize:
        #     voxels[x_i][y_i][z_i+1] += point["peakVal"]/2
        # if z_i > voxelSize:
        #     voxels[x_i][y_i][z_i-1] += point["peakVal"]/2

    return voxels

#
# Helper key function to sort frames
#
def key(s):
    if s[-7] in '0123456789':
        return int(s[-7:-4])
    if s[-6] in '0123456789':
        return int(s[-6:-4])
    else:
         return int(s[-5:-4])

#
# Function which can be used to plot objects if desired.
#
def plot_obj(objects):
    fig = plt.figure()
    x,y=[],[]
    for i in range(len(objects)):
        x.append(objects[i]["dopplerIdx"])
        y.append(objects[i]["rangeIdx"])
    plt.scatter(x,y)
    plt.show()

#
# Function which can be used to plot points.
#
def plot_points(objects):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x,y,z=[],[],[]
    vals =[]
    seen = []
    for i in range(len(objects)):
        if [objects[i]["x_coor"],objects[i]["y_coor"],objects[i]["z_coor"]] in seen :
            print("Seen")
        x.append(objects[i]["x_coor"])
        y.append(objects[i]["y_coor"])
        z.append(objects[i]["z_coor"])
        vals.append(objects[i]["peakVal"])
        seen.append([x[i],y[i],z[i]])
    ax.scatter(x, y, z, c=vals, alpha=1)
    plt.show()

#
# Function which can be used to plot the voxel if desired
#
def plot_voxel(voxel):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = voxel.nonzero()

    vals = [voxel[x[i],y[i],z[i]] for i in range(len(x))]
    ax.scatter(x, y, z, c=vals, alpha=1)
    plt.show()

#
# Function To verify if the Cluster can be used or we should skip this frame.
#
def cluster_ok(cluster,max_x,max_y,max_z,min_x,min_y,min_z):
    # Value chosen arbitrarly
    if len(cluster) < 4: # Note : Changed.. was 3....
        return False
    else:
        return (max_x > min_x) and (max_y > min_y) and (max_z > min_z)


#
# Function to get min and max values of x,y,z on the grouped points.
# Used to have a voxel with values spanning from min to max.
#
def getMinMax(points):
    max_x, max_y, max_z, min_x, min_y, min_z = points[0]["x_coor"],points[0]["y_coor"],points[0]["z_coor"],points[0]["x_coor"],points[0]["y_coor"],points[0]["z_coor"]#0,0,0,10,10,10
    for point in points:
        x_coor,y_coor,z_coor = point["x_coor"],point["y_coor"],point["z_coor"]
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
    return max_x, max_y, max_z, min_x, min_y, min_z

#
# Function essentially used to format structure correctly for convex hull interpolation
#
def toList(points):
    res = []
    res_val =[]
    for point in points: #zip(x_cords, y_cords, z_cords):
        res.append([point["x_coor"] ,point["y_coor"] ,point["z_coor"] ])
        res_val.append([point["x_coor"], point["y_coor"], point["z_coor"], point["peakVal"]])
    return np.array(res), np.array(res_val)

#
# Function for interpolation on the convex hull.
#
def sampleHull(hull,listHull,dict):
    vertices = listHull[hull.vertices]
    points = vertices

    for i in range(len(vertices) - 1):
        pt1,pt2 = points[i],points[i+1]
        step = (pt2-pt1)/6
        for i in range(1,5):
            pt3 = pt1 + i*step
            dict.append({"objId": -1, "peakVal": pt3[3],
                                "x_coor": pt3[0], "y_coor": pt3[1], "z_coor": pt3[2]})
    return dict

#
# Function to voxelise a 3d point cloud where points have previously been scaled to [-1,1]
#
def toVoxel_simple(points,voxelSize,max_x,max_y,max_z,min_x,min_y,min_z):
    voxels = np.zeros((voxelSize+1,voxelSize+1,voxelSize+1))

    seen = []
    for point in points: #zip(x_cords, y_cords, z_cords):
        x_i = int(voxelSize*(point["x_coor"] - min_x) / (max_x - min_x))
        y_i = int(voxelSize*(point["y_coor"] - min_y) / (max_y - min_y))
        z_i = int(voxelSize*(point["z_coor"] - min_z) / (max_z - min_z))
        if [x_i,y_i,z_i] not in seen:
            voxels[x_i][y_i][z_i] += point["peakVal"]
            seen.append([x_i,y_i,z_i])

    return voxels


#
# Main Loop, needs path as argument.
#
if __name__ == "__main__":
    path = sys.argv[1]

    print(listdir(path))
    files = listdir(path)
    files.sort(key=key)
    for fileName in files:

        n = path + "/" + fileName
        if not ".DS" in n:
            print("_______________________________________")
            print(fileName)
            print("_______________________________________")

            objects,max_x,max_y,max_z,min_x,min_y,min_z = read_detected_points(n)
            plot_points(objects)

            res = groupPoints(objects)
            plot_points(res)
            if cluster_ok(res,max_x,max_y,max_z,min_x,min_y,min_z):
                max_x, max_y, max_z, min_x, min_y, min_z = getMinMax(res)
                voxel = toVoxel(res,25,max_x,max_y,max_z,min_x,min_y,min_z)
                plot_voxel(voxel)

                voxel1 = toVoxel_simple(res, 25, max_x, max_y, max_z, min_x, min_y, min_z)
                plot_voxel(voxel1)
