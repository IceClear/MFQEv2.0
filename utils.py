import os, sys
sys.path.append('./')
import subprocess
import imageio
import numpy as np
import subprocess
import cv2
import glob
import math

def createpath(path):
    while not os.path.exists(path):
        os.makedirs(path)

def read_YUV420(image_path, rows, cols, numfrm):
    """
    读取YUV文件，解析为Y, U, V图像
    :param image_path: YUV图像路径
    :param rows: 给定高
    :param cols: 给定宽
    :return: 列表，[Y, U, V]
    """
    # create Y
    gray = np.zeros((rows, cols), np.uint8)
    # print(type(gray))
    # print(gray.shape)

    # create U,V
    img_U = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    # print(type(img_U))
    # print(img_U.shape)

    img_V = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    # print(type(img_V))
    # print(img_V.shape)
    Y = []
    U = []
    V = []
    reader=open(image_path,'rb')

    # with open(image_path, 'rb') as reader:
    for num in range(numfrm-1):
        Y_buf = reader.read(cols * rows)
        gray = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [rows, cols])

        U_buf = reader.read(cols//2 * rows//2)
        img_U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [rows//2, cols//2])

        V_buf = reader.read(cols//2 * rows//2)
        img_V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [rows//2, cols//2])

        Y = Y+[gray]
        U = U+[img_U]
        V = V+[img_V]

    return [Y, U, V]

def yuv2rgb(Y,U,V):

    enlarge_U = cv2.resize(U, (0, 0), fx=2.0, fy=2.0)
    enlarge_V = cv2.resize(V, (0, 0), fx=2.0, fy=2.0)

    # 合并YUV3通道
    img_YUV = cv2.merge([Y, enlarge_U, enlarge_V])

    dst = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)

    return dst

def saveimg(dir_saveframe,video_name,width,height,enhanced_frame,u_lq,v_lq,index):
    frame_to_save = np.squeeze(enhanced_frame)*255
    with open(dir_saveframe+video_name+'/frame_'+str(index) + '.yuv', 'wb') as fid:
        fid.write(frame_to_save.astype(np.uint8))
        fid.write(u_lq[index].astype(np.uint8))
        fid.write(v_lq[index].astype(np.uint8))
    subprocess.call([
        'ffmpeg', '-s',
        str(width) + '*' + str(height), '-pix_fmt', 'yuv420p', '-i',
        dir_saveframe+video_name+'/frame_'+str(index) + '.yuv', '-pix_fmt', 'rgb24', dir_saveframe+video_name+'/frame_'+str(index) + '.png'
    ])
    os.remove(dir_saveframe+video_name+'/frame_'+str(index) + '.yuv')
