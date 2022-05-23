import base64
import glob
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
import json
import pandas as pd
import os


# xml to csv
def xml2csv(xml_path):
    """Convert XML to CSV

    Args:
        xml_path (str): Location of annotated XML file
    Returns:
        pd.DataFrame: converted csv file

    """
    # print("xml to csv {}".format(xml_path))
    xml_list = []
    xml_df = pd.DataFrame()
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
            column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            xml_df = pd.DataFrame(xml_list, columns=column_name)
    except Exception as e:
        print('xml conversion failed:{}'.format(e))
        return pd.DataFrame(columns=['filename,width,height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    return xml_df


def df2labelme_459(xml_df, img_path, image):
    h, w = image.shape[:2]
    with open(img_path, 'rb') as f:
        image = f.read()
    image_base64 = str(base64.b64encode(image), encoding='utf-8')

    version = "4.5.9"
    data_dict = dict()
    data_dict.__setitem__("version", version)
    data_dict.__setitem__("flags", {})
    data_dict["shapes"] = list()
    for class_name, x1, y1, x2, y2 in zip(xml_df['class'], xml_df['xmin'], xml_df['ymin'], xml_df['xmax'], xml_df['ymax']):
        points = [[x1, y1], [x2, y2]]
        shape_type = "rectangle"
        shape = {}

        shape.__setitem__("label", class_name)
        shape.__setitem__("points", points)
        shape.__setitem__("group_id", None)
        shape.__setitem__("shape_type", shape_type)
        shape.__setitem__("flags", {})
        data_dict["shapes"].append(shape)
    data_dict.__setitem__("imagePath", img_path)
    data_dict.__setitem__("imageData", image_base64)
    data_dict.__setitem__("imageHeight", h)
    data_dict.__setitem__("imageWidth", w)
    # print(data_dict)
    return data_dict


def main():
    xml_dir = '/home/yib11/VOC2028/Annotations'
    xml_path_list = glob.glob(os.path.join(xml_dir, "*.xml"))
    xml_path_list.sort()

    for xml_path in tqdm(xml_path_list):
        xml_csv = xml2csv(xml_path)
        image_path = xml_path.replace('Annotations', 'JPEGImages').replace('xml', 'jpg')
        image = cv2.imread(image_path)
        csv_json = df2labelme_459(xml_csv, image_path, image)
        json_path = xml_path.replace('Annotations', 'HatAnnotations').replace('xml', 'json')
        with open(json_path, "w") as f:
            json.dump(csv_json, f, indent=2)
        # print("载入文件", json_path)


if __name__ == '__main__':
    main()
