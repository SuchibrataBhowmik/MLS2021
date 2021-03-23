
import xml.etree.ElementTree as et
import csv

def read_xml(file):
    tree = et.parse(file)
    root = tree.getroot() 

    d =[]
    for child in root[2:]:
        for in_child in child:
             d.append([ int(in_child.attrib['frame']), int(float(in_child.attrib['xtl'])), int(float(in_child.attrib['ytl'])),
                       int(float(in_child.attrib['xbr'])), int(float(in_child.attrib['ybr'])), child.attrib['label'], int(in_child.attrib['outside']) ])
    
    return d

def write_csv(data):
    header = ['frameno','left','top','right','bottom','label','outside']
    
    with open('track.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)


data = read_xml("track.xml")
write_csv(data)
print("track.csv file successfully save")
