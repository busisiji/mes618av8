import logging
import os
import glob
import xml.etree.ElementTree as ET
import yaml

def convert(box,dw,dh):
    x=(box[0]+box[2])/2.0
    y=(box[1]+box[3])/2.0
    w=box[2]-box[0]
    h=box[3]-box[1]

    x=x/dw
    y=y/dh
    w=w/dw
    h=h/dh

    return x,y,w,h

def convert_annotation(name_id,classes,xml_file,txt_file):
    xml_o=open(xml_file+r'\%s.xml'%name_id, encoding='UTF-8')
    txt_o=open(txt_file+r'\%s.txt'%name_id,'w',encoding='UTF-8')

    pares=ET.parse(xml_o)
    root=pares.getroot()
    objects=root.findall('object')
    size=root.find('size')
    dw=int(size.find('width').text)
    dh=int(size.find('height').text)

    for obj in objects :
        try:
            c=classes.index(obj.find('name').text)
            bnd=obj.find('bndbox')

            b=(float(bnd.find('xmin').text),float(bnd.find('ymin').text),
               float(bnd.find('xmax').text),float(bnd.find('ymax').text))

            x,y,w,h=convert(b,dw,dh)

            write_t="{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(c,x,y,w,h)
            txt_o.write(write_t)
        except Exception as e:
            logging.exception("【xml转txt文件报错】")

    xml_o.close()
    txt_o.close()

if __name__ == '__main__':
    xml_file = r'../data/Annotations'
    txt_file = r'../data/labels'
    # classes = ['unpass',
    #      'toolong',
    #      'bad',
    #      'transparentempty',
    #      'transparenexcept',
    #      'withoutcover',
    #      'havecover',
    #      'none',
    #      'XLabel',
    #      'YLabel', ]
    with open('data/618A.yaml', 'r', encoding='utf-8') as f:
        data_yaml = yaml.load(f, Loader=yaml.FullLoader)
    class_names = []  # 标签类别
    for value in data_yaml['names'].values():
        class_names.append(value)
    name=glob.glob(os.path.join(xml_file,"*.xml"))
    for i in name:
        name_id=os.path.basename(i)[:-4]
        convert_annotation(name_id,class_names)
