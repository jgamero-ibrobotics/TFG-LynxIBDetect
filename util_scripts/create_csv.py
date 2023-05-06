# Script to create CSV data file from Pascal VOC annotation files
# Based off code from GitHub user datitran: https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            filename = root.find('filename').text
            width = int(root.find('size')[0].text)
            height = int(root.find('size')[1].text)
            class_name = member[0].text
            xmin, ymin, xmax, ymax = None, None, None, None

            # Verifica si el objeto tiene una etiqueta de caja delimitadora
            if member.find('bndbox') is not None:
                xmin = int(member.find('bndbox/xmin').text)
                ymin = int(member.find('bndbox/ymin').text)
                xmax = int(member.find('bndbox/xmax').text)
                ymax = int(member.find('bndbox/ymax').text)

            value = (filename, width, height, class_name, xmin, ymin, xmax, ymax)
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train','validation']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/' + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')

main()
