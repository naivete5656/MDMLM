import xml.etree.ElementTree as ET
import cv2
import numpy as np

tree = ET.parse(f'/home/kazuya/Downloads/itousensei_annotation.xml')
root = tree.getroot()
parent_cell_cands = root.findall('.//a')

annotations = []
for a in parent_cell_cands:
    if len(a) == 2:
        length = len(a.find('ss'))
        t = a.find('ss')[length - 1].attrib['i']
        x = a.find('ss')[length - 1].attrib['x']
        y = a.find('ss')[length - 1].attrib['y']
        annotations.append([int(float(x)), int(float(y)),int(t)])

annotations = np.array(annotations)

np.savetxt("mitosis_list.txt", annotations)
