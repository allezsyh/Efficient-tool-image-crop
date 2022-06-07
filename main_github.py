# coding=utf-8
import time

import cv2
import os
import threading
from threading import Lock, Thread
import glob

video_path = "E:/dataset/thumos14"
pic_path = "E:/dataset/thumos14/pics"
filelist = glob.glob(os.path.join(video_path, '*.avi'))
filelist.sort()


def video2pic(filename):
    # print(filename)
    video_name = filename.split('.')[0]
    cnt = 0
    dnt = 0
    if not os.path.exists(os.path.join(pic_path, video_name)):
        os.makedirs(os.path.join(pic_path, video_name))
    cap = cv2.VideoCapture(os.path.join(video_path, filename))  # 读入视频
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] {} total frames in video {}".format(total, video_name))
    t1 = time.time()
    while True:
        # get a frame
        ret, image = cap.read()
        if image is None:
            break
        # show a frame
        w = image.shape[1]
        h = image.shape[0]
        group_path = os.path.join(pic_path, str(video_name))
        if not os.path.exists(group_path):
            os.mkdir(group_path)
        dnt = dnt + 1
        cv2.imencode('.jpg', image)[1].tofile(group_path + '/' + str(video_name) + '_' + str(dnt).zfill(7) + '.jpg')
        t2 = time.time()
        dt = (t2 - t1) * 1E3
        t1 = t2
        print('[' + str(cnt) + '/' + str(total) + '] ' + group_path + '/' + str(video_name) + '_' + str(dnt).zfill(7) + '.jpg' + ' [' + str(dt) + 'ms]')
        cnt = cnt + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    for filename in filelist:
        filename = filename.split(os.sep)[-1]
        video2pic(filename)