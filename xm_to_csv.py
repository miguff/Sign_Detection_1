import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

directory = "/home/miguff/ÓE/Sign_Recognition/dataset/annotations"

xml_lista = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    tree = ET.parse(f)
    root = tree.getroot()
    for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(bbx.find("xmin").text)
            ymin = int(bbx.find("ymin").text)
            xmax = int(bbx.find("xmax").text)
            ymax = int(bbx.find("ymax").text)
            label = member.find('name').text

            érték = (root.find('filename').text,
                     xmin,
                     ymin,
                     xmax,
                     ymax,
                     label
                     )
            xml_lista.append(érték)
            xml_df = pd.DataFrame(xml_lista, columns=None)

xml_df.to_csv('/home/miguff/ÓE/Sign_Recognition/dataset/annotations/annotations.csv', index=None, header=None)