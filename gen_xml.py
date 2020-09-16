import os
from os import getcwd
from xml.etree import ElementTree as ET
# from lxml import etree as ET
from detect_1 import *


# 定义一个创建一级分支object的函数
def create_object(root, xi, yi, xa, ya, obj_name):  # 参数依次，树根，xmin，ymin，xmax，ymax
    # 创建一级分支object
    _object = ET.SubElement(root, 'object')
    # 创建二级分支
    name = ET.SubElement(_object, 'name')
    # print(obj_name)
    name.text = str(obj_name)
    pose = ET.SubElement(_object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(_object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(_object, 'difficult')
    difficult.text = '0'
    # 创建bndbox
    bndbox = ET.SubElement(_object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '%s' % xi
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '%s' % yi
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '%s' % xa
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '%s' % ya


# 创建xml文件的函数
def create_tree(image_name, h, w, imgdir):
    global annotation
    # 创建树根annotation
    annotation = ET.Element('annotation')
    # 创建一级分支folder
    folder = ET.SubElement(annotation, 'folder')
    # 添加folder标签内容
    folder.text = (imgdir)

    # 创建一级分支filename
    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_name

    # 创建一级分支path
    path = ET.SubElement(annotation, 'path')

    path.text = getcwd() + '\{}\{}'.format(imgdir, image_name)  # 用于返回当前工作目录

    # 创建一级分支source
    source = ET.SubElement(annotation, 'source')
    # 创建source下的二级分支database
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    # 创建一级分支size
    size = ET.SubElement(annotation, 'size')
    # 创建size下的二级分支图像的宽、高及depth
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    # 创建一级分支segmented
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'


def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


if __name__ == '__main__':
    # Load yolov3_tiny_se detect
    weights = 'yolov5s.pt'
    device = torch_utils.select_device(device='0')
    half = device.type != 'cpu'

    model = torch.load(weights, map_location=device)['model'].float()
    model.to(device).eval()
    if half:
        model.half()
    imgdir = 'images'
    outdir = 'Annotation'
    names = model.module.names if hasattr(model, 'module') else model.names
    IMAGES_LIST = os.listdir(imgdir)
    for image_name in IMAGES_LIST:
        # print(image_name)
        # 判断后缀只处理jpg文件
        if image_name.endswith('.jpg'):
            image = cv2.imread(os.path.join(imgdir, image_name))
            coordinates_list = detector(image, model, device, half)
            (h, w) = image.shape[:2]
            create_tree(image_name, h, w, imgdir)
            if coordinates_list:
                print(image_name)
                for coordinate in coordinates_list:
                    label_id = coordinate[4]
                    create_object(annotation, int(coordinate[0]), int(coordinate[1]), int(coordinate[2]), int(coordinate[3]), names[label_id])

                # 将树模型写入xml文件
                tree = ET.ElementTree(annotation)
                root = tree.getroot()
                pretty_xml(root, '\t', '\n')
                tree.write('./{}/{}.xml'.format(outdir, image_name.strip('.jpg')), encoding='utf-8')
            else:
                print(image_name)