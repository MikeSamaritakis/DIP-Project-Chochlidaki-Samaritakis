# convert_xml_to_json.py
import sys
import xmltodict
import json
import numpy as np

def convert_xml_to_json(xml_path, output_path):
    print("Converting xml to json")
    with open(xml_path) as fd:
        doc = xmltodict.parse(fd.read())

        data_list = []
        for anno in doc["annotations"]["image"]:
            data = {}
            data['width'] = int(anno['@width'])
            data['height'] = int(anno['@height'])
            data['name'] = anno['@name']
            data['stitches'] = []
            data['incisions'] = []
            if "polyline" in anno:
                if not isinstance(anno["polyline"], list):
                    pts = [[float(x) for x in pt.split(",")] for pt in anno["polyline"]["@points"].split(";")]
                    if pline['@label'] == 'Incision':
                        data['incisions'].append(pts)
                    if pline['@label'] == 'Stitch':
                        data['stitches'].append(pts)

                else:
                    for pline in anno["polyline"]:
                        pts = [[float(x) for x in pt.split(",")] for pt in pline["@points"].split(";")]
                        if pline['@label'] == 'Incision':
                            data['incisions'].append(pts)
                        if pline['@label'] == 'Stitch':
                            data['stitches'].append(pts)

            data_list.append(data)

        with open(output_path,'w') as fw:
            json.dump(data_list, fw)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_xml_to_json.py input.xml output.json")
        sys.exit(1)
    convert_xml_to_json(sys.argv[1], sys.argv[2])
