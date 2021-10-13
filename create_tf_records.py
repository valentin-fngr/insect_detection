import os 
import tensorflow as tf  
import numpy as np 
import matplotlib.pyplot as plt 
import json

DATASET_ROOT = os.path.join(os.getcwd(), "dataset/ArTaxOr")

labels = ["Coleoptera", "Diptera", "Hemiptera", "Hymenoptera", "Lepidoptera", "Odonata"]
tfrecords_dir = os.path.join(os.getcwd(), "dataset/records")

def get_label_by_id(id):
    """
        simply return the corresponding label, given an id
    """
    return labels[id]


def serialize_sample(label_path): 

    # read file 
    with open(label_path) as f: 
        data = json.load(f)
    
    image_path = "/".join(data["asset"]["path"].split("/")[-2:]) 
    image_path = os.path.join(DATASET_ROOT, image_path)

    img = tf.io.read_file(image_path) 
    img = tf.image.decode_png(img)

    regions = data["regions"]

    width = data["asset"]["size"]["width"]
    height = data["asset"]["size"]["height"]

    img = tf.image.resize(img, (448,448,3))
    img = tf.cast(img, dtype=tf.uint8)
    encoded_image = tf.io.encode_jpeg(img).numpy() # encode to bytes

    obj_labels = [] 
    heights = [] 
    widths = [] 
    lefts = []  
    tops = []


    for region in regions: 

        heights.append(int(region["boundingBox"]["height"]) / 448) 
        widths.append(int(region["boundingBox"]["width"]) / 448) 
        lefts.append(int(region["boundingBox"]["left"]) / 448)
        tops.append(int(region["boundingBox"]["top"]) / 448)
        obj_labels.append(labels.index(region["tags"][0]))

    feature = {
        "image/encoded" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
        "image/obj/heights": tf.train.Feature(int64_list=tf.train.Int64List(value=heights)), 
        "image/obj/widths": tf.train.Feature(int64_list=tf.train.Int64List(value=widths)),
        "image/obj/lefts": tf.train.Feature(int64_list=tf.train.Int64List(value=tops)),
        "image/obj/tops": tf.train.Feature(int64_list=tf.train.Int64List(value=lefts)), 
        "image/obj/class_id": tf.train.Feature(int64_list=tf.train.Int64List(value=obj_labels)), 
    }

    features = tf.train.Features(feature=feature)

    example = tf.train.Example(features=features)
    return example.SerializeToString()


def create_tf_records(annot_path_list, label, num_sample=200, tfrecords_dir=tfrecords_dir): 

    # calculate the number of tf records 
    num_tfrecords = len(annot_path_list) // num_sample
    if len(annot_path_list) % num_sample != 0: 
        num_tfrecords += 1
    
    if not os.path.exists(tfrecords_dir): 
        os.makedirs(tfrecords_dir)

    for i in range(num_tfrecords): 
        sample_path = annot_path_list[i*num_sample:(i+1)*num_sample]

        with tf.io.TFRecordWriter(f'{tfrecords_dir}/file_{i}_{len(sample_path)}_{label}.tfrecord') as writer:
            removed = 0
            for sample in sample_path:
                try:
                    serialized_example = serialize_sample(sample) 
                    writer.write(serialized_example)
                except (ValueError, Exception): 
                    print("couldn't serialize example for image :", sample)
                    removed += 1
            print("couldn't load", removed)
            print(f"Successfuly created tf record file file_{i}_{len(sample_path) - removed}_{label}.tfrecords")
            print()


if __name__ == "__main__": 
    
    label_path = []

    for cls in labels: 
        cls_path = os.path.join(DATASET_ROOT, cls)
        annot_path = os.path.join(cls_path, "annotations")
        annot_path_list = [os.path.join(annot_path, file_path) for file_path in  os.listdir(annot_path)]
        
        create_tf_records(annot_path_list, label=cls)
        
            
