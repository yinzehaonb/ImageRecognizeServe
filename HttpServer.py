from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flask import Flask,request
import json
import numpy as np
import tensorflow as tf
import os
from hashlib import md5
label_path = 'C:\\Users\\PC\\Documents\\Tencent Files\\654808268\\FileRecv\\plant_labels.txt'
graph_path = 'C:\\Users\\PC\\Documents\\Tencent Files\\654808268\\FileRecv\\inception-resnet-v2-2000000.pb'

class NodeLookup(object):
    def __init__(self, label_lookup_path=None):
        self.node_lookup = self.load(label_lookup_path)

    def load(self, label_lookup_path):
        node_id_to_name = {}
        with open(label_lookup_path,encoding='utf-8') as f:
            for index, line in enumerate(f):
                node_id_to_name[index] = line.strip()
        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph():
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def run_inference_on_image(image_data,sess):

    softmax_tensor = sess.graph.get_tensor_by_name('InceptionResnetV2/Logits/Predictions:0')
    predictions = sess.run(softmax_tensor,
                           {'input:0': image_data})
    predictions = np.squeeze(predictions)
    node_lookup = NodeLookup(label_path)

    top_k = predictions.argsort()[-5:][::-1]
    print(top_k)
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))
    return [(node_lookup.id_to_string(node_id),predictions[node_id].item()) for node_id in top_k]

create_graph()
sess = tf.Session()

app = Flask(__name__)
@app.route('/',methods=['POST'])
def main():

    try:
        data = request.files.get('data').read()
        fmd5 = md5(data)
        fmd5 = str(fmd5.hexdigest())
        dir = os.path.join('E:/pic/test',fmd5+'.jpg')
        # 保存数据
        pic = open(dir,'wb')
        pic.write(data)
        pic.close()
        # 加载数据到tensor
        image_data = tf.gfile.FastGFile(dir, 'rb').read()
        # print(image_data,type(image_data))
        # 删除缓存数据
        os.remove(dir)
        g = tf.Graph()
        with g.as_default():
            # 创建第二个计算图，隔离图
            image_data = tf.image.decode_jpeg(image_data)
            image_data = preprocess_for_eval(image_data, 299, 299)
            image_data = tf.expand_dims(image_data, 0)
        with tf.Session(graph=g) as sess2:
            # 第二个图的对应的SESS
            image_data = sess2.run(image_data)

        result = run_inference_on_image(image_data,sess)
        res = []
        for item in result:
            res_dic = {}
            # print(i[0].split(' ')[-1])
            res_dic['score'] = "%.8f" % item[1]
            res_dic['name'] = item[0].split(' ')[-1]
            sorted(res_dic.items(), key=lambda item: item[0], reverse=False)
            res.append(res_dic)
        return json.dumps(res)
        # return json.dumps(res)
    except Exception as e:
        print(e)
        return repr(e),500
# app.run(host='0.0.0.0',port = 12480)
app.run(host='127.0.0.1',port = 12480)

