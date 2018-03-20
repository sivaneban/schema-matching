import sys
from xml.etree import ElementTree

import numpy as np
import pandas as pd


def pandas_df_to_xml(df, field_names, root_tag='items', element_tag='item'):
    root = ElementTree.Element(root_tag)
    for i, row in df.iterrows():
        item = ElementTree.SubElement(root, element_tag)
        for field in field_names:
            sub_element = ElementTree.SubElement(item, field)
            sub_element.text = str(row[field])
    xml_string = ElementTree.tostring(root, encoding='utf-8')
    return xml_string

def main():
    argv = sys.argv
    splits = argv[3]
    inputfile = argv[1]
    outputfile = argv[2]
    
    print("Inputfile: " + inputfile)
    print("Outputfile: " + outputfile)
    print("Splits: " + splits)
    
    splits = int(splits)
    df = pd.read_csv(inputfile)
    dfs = np.array_split(df, splits)
    
    for i in range(splits):
        if splits == 1:
           outname = outputfile
        else:
           outname = outputfile.split('.')[0] + '_' + str(i + 1) + '.xml'
        xml = pandas_df_to_xml(dfs[i], dfs[i].columns.values.tolist())
        with open(outname, 'wb') as outfile:
            outfile.write(xml)
    

if __name__ == '__main__':
    main()