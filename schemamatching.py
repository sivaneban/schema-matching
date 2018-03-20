import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

def create_pipeline():
    count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2), lowercase=False)
    tree_classifier = DecisionTreeClassifier()
    pipeline = Pipeline([
        ('vectorizer', count_vectorizer), 
        ('classifier' , tree_classifier)
    ])
    return pipeline

def collect_instance_data(xml_string):
    def _recurse(data, current_path, xml_element):
        current_path = current_path + '/' + xml_element.tag
        if len(xml_element) == 0:
            if current_path not in data:
                data[current_path] = []
            data[current_path].append(xml_element.text)
        else:    
            for child in xml_element:
                _recurse(data, current_path, child)
    data = {}
    xml_tree = ET.fromstring(xml_string)
    _recurse(data, '', xml_tree)
    return data

def get_features(xml):
    data_as_dict = collect_instance_data(xml)
    df = pd.DataFrame(data_as_dict)
    df = df.melt(var_name='tag', value_name='item')
    return df

def compare_xmls(xml1, xml2):
    xml1_data = collect_instance_data(xml1)
    xml1_features = get_features(xml1)
    xml2_data = collect_instance_data(xml2)
    pipeline = create_pipeline()
    pipeline.fit(xml1_features['item'], xml1_features['tag'])

    output_shape =len(xml2_data.keys()), len(xml1_data.keys())
    outputs = pd.DataFrame(np.zeros(output_shape),
        index=xml2_data.keys(), columns=xml1_data.keys())
    
    for tag in xml2_data:
        predictions = pipeline.predict(xml2_data[tag])
        total = len(predictions)
        for p in predictions:
            outputs.loc[tag, p] += 1.0 / total
    return outputs
