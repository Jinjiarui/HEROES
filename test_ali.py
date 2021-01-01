import glob
import os
import random
from collections import defaultdict
import pickle
import numpy as np
import tensorflow as tf
from hdfs.client import Client

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
client = Client("http://localhost:9870", root='cxy')
"""
df = pd.read_csv('temp2/sample_skeleton_test.csv',nrows=100,encoding='utf8')
print(df.head())
"""


def read_part(inf):
    doc_num = 0
    query_num = 0
    click_num = 0
    conversion_num = 0
    query = None
    for file in inf:
        with open(file, encoding='utf8', mode='r') as fin:
            line = fin.readline()
            while line:
                line = line.rstrip().split()
                if line[0] != query:
                    query = line[0]
                    query_num += 1
                doc_num += 1
                click_num += int(line[2])
                conversion_num += int(line[3])
                line = fin.readline()
    print(
        "doc_num:{}\tquery_num:{}\tclick_num:{}\tconversion_num:{}".format(doc_num, query_num, click_num,
                                                                           conversion_num))


def read_user(inf):
    user1 = set()
    user2 = set()
    item1 = set()
    item2 = set()
    with open(inf[0], encoding='utf8', mode='r') as fin:
        line = fin.readline()
        while line:
            line = line.rstrip().split()
            user1.add(int(line[4]))
            item1.add(int(line[5]))
            line = fin.readline()
    with open(inf[1], encoding='utf8', mode='r') as fin:
        line = fin.readline()
        while line:
            line = line.rstrip().split()
            user2.add(int(line[4]))
            item2.add(int(line[5]))
            line = fin.readline()
    print(len(user1 - user2), len(item1 - item2))


def load_fcnts(if_str, out_path):
    feat_cnts_dict = defaultdict(lambda: 0)
    feat_cnts_float = defaultdict(lambda: 0)
    user_id_dict = defaultdict(lambda: 0)
    item_id_dict = defaultdict(lambda: 0)
    float_dict = ['109_14', '110_14', '127_14', '150_14', '508', '509', '702', '853']
    new_id = 0
    float_id = 0
    user_id = 0
    item_id = 0
    with open(if_str, 'r', encoding='utf8') as f:
        for line in f:
            fid, cnts = line.strip().split('\t')
            if int(cnts) >= 100:  # cutoff=100
                if fid.split(":")[0] in float_dict:
                    feat_cnts_float[fid] = float_id
                    float_id = float_id + 1
                else:
                    feat_cnts_dict[fid] = new_id
                    new_id = new_id + 1
            if fid.split(":")[0] == '101':  # 用户id
                user_id_dict[fid.split(":")[1]] = user_id
                user_id += 1
            if fid.split(":")[0] == '205':  # 商品id
                item_id_dict[fid.split(":")[1]] = item_id
                item_id += 1

    with open(os.path.join(out_path, 'r_down_remap.txt'), 'w', encoding='utf8') as f:
        for k, v in feat_cnts_dict.items():
            f.write("{0}\t{1}\n".format(k, v))
            print(k, v)
        for k, v in feat_cnts_float.items():
            f.write("{0}\t{1}\n".format(k, v + len(feat_cnts_dict)))
            print(k, v)
    with open(os.path.join(out_path, 'r_down_user_remap.txt'), 'w', encoding='utf8') as f:
        for k, v in user_id_dict.items():
            f.write("{0}\t{1}\n".format(k, v))
            print(k, v)
    with open(os.path.join(out_path, 'r_down_item_remap.txt'), 'w', encoding='utf8') as f:
        for k, v in item_id_dict.items():
            f.write("{0}\t{1}\n".format(k, v))
            print(k, v)


def get_cnts_distribution(if_str):
    feat_cnts_dict = defaultdict(lambda: 0)
    with open(if_str, 'r', encoding='utf8') as f:
        # print(len(f.readlines()))
        for line in f:
            fid, cnts = line.strip().split('\t')
            '''if fid.split(':')[0] == '101':
                feat_cnts_dict[fid] = int(cnts)'''
            feat_cnts_dict[fid] = int(cnts)
    for i in [1, 10, 100, 1000, 10000, 100000, 1000000]:
        print(np.sum(list(map(lambda x: x >= i, feat_cnts_dict.values()))))


def read_test(inf):
    def _parse_fn(record):
        features = {
            "y": tf.FixedLenFeature([], tf.float32),
            "z": tf.FixedLenFeature([], tf.float32),
            "user_id": tf.FixedLenFeature([], tf.int64),
            "item_id": tf.FixedLenFeature([], tf.int64),
            "other_feature_id": tf.VarLenFeature(tf.int64),
            "other_feature_val": tf.VarLenFeature(tf.float32),
        }
        parsed = tf.parse_single_example(record, features)
        y = parsed.pop('y')
        z = parsed.pop('z')
        return parsed, {"y": y, "z": z}

    dataset = tf.data.TFRecordDataset(inf)
    dataset = dataset.map(_parse_fn, num_parallel_calls=64).prefetch(50000)
    dataset = dataset.batch(4000)
    item = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        while True:
            try:
                batch_features, batch_labels = sess.run(item.get_next())
                y, z = batch_labels['y'], batch_labels['z']
                epsilon = 1e-7
                click_num = tf.to_float(tf.count_nonzero(y))
                conversion_num = tf.to_float(tf.count_nonzero(z))
                click_weight = (tf.to_float(tf.size(y)) - click_num) / (click_num + epsilon)
                conversion_weight = (tf.to_float(tf.size(z)) - conversion_num) / (conversion_num + epsilon)
                print(sess.run(click_weight), sess.run(conversion_weight))
            except Exception as e:
                print(e)
                break


def native_down_lstm(infile, outfile):
    common_dir = defaultdict(lambda: [])
    click_dir = defaultdict(lambda: 0)
    conversion_dir = defaultdict(lambda: 0)
    common_len_dir = defaultdict(lambda: 0)
    num = 0
    with open(infile) as inf:
        old_id = None
        old_conversion = False
        for line in inf:
            splits = line.strip().split(',')
            if splits[3] != old_id:
                old_id = splits[3]
                if old_conversion:
                    num += 1  # 检测发生在最后一次浏览的转换
            if splits[1] == '0' and splits[2] == '1':
                continue
            common_dir[splits[3]].append(line)
            if splits[1] == '1':
                click_dir[splits[3]] += 1
            if splits[2] == '1':
                conversion_dir[splits[3]] += 1
                old_conversion = True
            else:
                old_conversion = False
            common_len_dir[splits[3]] += 1
    conversion_common = list(
        filter(lambda k: common_len_dir[k] >= 3, conversion_dir.keys()))
    print(len(conversion_common), num)
    click_only_common = list(
        filter(lambda k: (conversion_dir[k] == 0 and common_len_dir[k] >= 3), click_dir.keys()))
    '''native_common = list(filter(lambda k: (common_len_dir[k] >= 3),
                                common_dir.keys() - (conversion_common + click_only_common)))
    random.shuffle(click_only_common)
    random.shuffle(native_common)
    print(len(conversion_common), len(click_only_common), len(native_common))
    click_only_common = click_only_common[:int(len(conversion_common))]
    native_common = native_common[:int(len(native_common) / 10)]'''
    with open(outfile, mode='w') as outf:
        for positive_common in conversion_common:
            for sample in common_dir[positive_common]:
                outf.write(sample)
        '''for click_only_sample in click_only_common:
            for sample in common_dir[click_only_sample]:
                outf.write(sample)
        for negative_sample in native_common:
            for sample in common_dir[negative_sample]:
                outf.write(sample)'''


def native_down(infile, outfile):
    conversion_positive_list = []
    click_only_positive_list = []
    negative_list = []
    cnt_num = 0
    with open(infile) as inf:
        line = inf.readline()
        while line:
            cnt_num += 1
            if cnt_num % 100000 == 0:
                print(cnt_num)
            splits = line.strip().split(',')
            if splits[1] == '0' and splits[2] == '1':
                line = inf.readline()
                continue
            elif splits[2] == '1':
                conversion_positive_list.append(line)
            elif splits[1] == '1':
                click_only_positive_list.append(line)
            else:
                negative_list.append(line)
            line = inf.readline()
    random.shuffle(click_only_positive_list)
    random.shuffle(negative_list)
    click_only_positive_list = click_only_positive_list[:int(len(conversion_positive_list) * 2.5)]
    negative_list = negative_list[:int(len(click_only_positive_list) * 2.5)]
    with open(outfile, mode='w') as outf:
        for positive_sample in conversion_positive_list:
            outf.write(positive_sample)
        for click_only_sample in click_only_positive_list:
            outf.write(click_only_sample)
        for negative_sample in negative_list:
            outf.write(negative_sample)


if __name__ == '__main__':
    in_files = glob.glob("alicpp/t*/remap*/r*.txt")
    print(in_files)
    out_files = [_ + 'native_down_lstm' for _ in in_files]
    print("in_files:", in_files)
    '''for i in range(len(in_files)):
        native_down_lstm(in_files[i], out_files[i])'''
    for in_file in in_files:
        click = 0
        conversion = 0
        feat_cnts_dict = defaultdict(lambda: 0)
        doc_num = 0
        with open(in_file) as f:
            seq_len_l = []
            old_common = None
            seq_len = 0
            for line in f:
                doc_num += 1
                splits = line.strip().split()
                click += int(splits[2])
                conversion += int(splits[3])
                feat_cnts_dict[splits[0]] += 1
                old_common = splits[0]
            print(click, conversion, len(feat_cnts_dict), doc_num)
            with open(in_file + '.pkl', 'rb') as len_f:
                data = list(pickle.load(len_f))
            data[-1] = feat_cnts_dict[old_common]
            with open(in_file + '.pkl', 'wb') as len_f:
                pickle.dump(data,len_f)
            with open(in_file + '.pkl', 'rb') as len_f:
                data = list(pickle.load(len_f))
            print(list(feat_cnts_dict.values())[-10:])
            print(data[-10:])
            print(list(feat_cnts_dict.values())==data)
            print(len(data))

            '''for i in range(1, 50):
                print(i, np.sum(list(map(lambda x: x >= i, feat_cnts_dict.values()))))'''
            # get_cnts_distribution('alicpp/cnts/r_down_cnt.txt')
            # load_fcnts('alicpp/cnts/r_down_cnt.txt', 'alicpp/cnts')
