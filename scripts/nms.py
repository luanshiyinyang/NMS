"""
Author: Zhou Chen
Date: 2020/4/30
Desc: Numpy实现nms
"""
import numpy as np
import cv2

from draw_bbox import draw_box


def nms(bboxes, scores, iou_thresh):
    """

    :param bboxes: 检测框列表
    :param scores: 置信度列表
    :param iou_thresh: IOU阈值
    :return:
    """

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    # 结果列表
    result = []
    index = scores.argsort()[::-1]  # 对检测框按照置信度进行从高到低的排序，并获取索引
    # 下面的操作为了安全，都是对索引处理
    while index.size > 0:
        # 当检测框不为空一直循环
        i = index[0]
        result.append(i)  # 将置信度最高的加入结果列表

        # 计算其他边界框与该边界框的IOU
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # 只保留满足IOU阈值的索引
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]  # 处理剩余的边框
    bboxes, scores = bboxes[result], scores[result]
    return bboxes, scores


if __name__ == '__main__':
    raw_img = cv2.imread('test.png')
    # 这里为了编码方便，将检测的结果直接作为变量
    bboxes = [[183, 625, 269, 865], [197, 603, 296, 853], [190, 579, 295, 864], [537, 507, 618, 713], [535, 523, 606, 687]]
    confidences = [0.7, 0.9, 0.95, 0.9, 0.6]
    # 未经过nms的原始检测结果
    img = raw_img.copy()
    for x, y in zip(bboxes, confidences):
        img = draw_box(img, x, y)
    cv2.imwrite("../assets/raw_img.png", img)
    # 进行nms处理
    bboxes, scores = nms(np.array(bboxes), np.array(confidences), 0.5)
    img = raw_img.copy()
    for x, y in zip(list(bboxes), list(scores)):
        img = draw_box(img, x, y)
    cv2.imwrite("../assets/img_nms.png", img)