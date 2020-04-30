import cv2


def draw_box(img, bbox, confidence=None, offset=(0, 0)):
    """
    图上绘制一个边界框
    :param img:
    :param bbox: 必须是(xmin, ymin, xmax, ymax)的格式
    :param confidence: 该边界框的置信度
    :param offset:
    :return:
    """
    x1, y1, x2, y2 = [int(i) for i in bbox]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[1]
    y2 += offset[1]
    # box text and bar
    color = 0.5
    label = str(confidence)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
    cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

