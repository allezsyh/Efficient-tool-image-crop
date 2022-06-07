import os
import cv2
import torch
import glob
import numpy as np

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = np.array(xyxy)
    xyxy = xyxy.reshape((1,xyxy.shape[0]))
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b)
    # clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    cv2.imwrite(file, crop)
    return crop

if __name__ == '__main__':
    img_root = ''
    save_path = ''
    # E:\dataset\ASD_rti\pictures_doctor\pic_07\0720211125_111250_2899.jpg
    xyxy = np.array([0, 0, 0, 0])
    vn_ls = os.listdir(img_root)
    first_frame = True
    for i, vn in enumerate(vn_ls):
        vn_path = os.path.join(img_root, vn)
        img_ls = os.listdir(vn_path)
        for j, img in enumerate(img_ls):
            if not os.path.isfile(os.path.join(vn_path, img)): continue
            im = cv2.imread(os.path.join(vn_path, img))
            if first_frame:
                cv2.namedWindow(img, cv2.WINDOW_AUTOSIZE)
                # cv2.moveWindow(img, 100, 100)
                im0 = im.copy()
                init_rect = cv2.selectROI(img, im0, False, False)
                x, y, w, h = init_rect
                first_frame = False
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                cv2.destroyAllWindows()
                xyxy = np.array([x, y, x+w, x+h])
                with open('img_crop_position', "w") as out_file:
                    out_file.write(str(x) + ' ' + str(y) + ' ' + str(x+w) + ' ' + str(x+h))

            if not os.path.exists(os.path.join(save_path, vn)):
                os.makedirs(os.path.join(save_path, vn))
            save_file = os.path.join(save_path, vn, img)
            crop = save_one_box(xyxy, im, save_file, BGR=True)
            print('[{}/{}][{}/{}] {} {}'.format(i, len(vn_ls), j, len(img_ls), save_file, crop.shape))
