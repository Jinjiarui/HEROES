import glob
import tensorflow as tf
from collections import defaultdict


def count_max(inf):
    fea_l = set()
    fea_str = set()
    num = 0
    doc_num = 0
    click_num = 0
    conversion_num = 0
    occur = defaultdict(lambda: 0)
    for file in inf:
        with open(file, encoding='utf8', mode='r') as fin:
            line = fin.readline()
            while line:
                (seq_len, label) = [int(_) for _ in line.split()]
                num += 1
                doc_num += seq_len
                occur[seq_len] += 1
                conversion_num += label
                for _ in range(seq_len):
                    tmpline = fin.readline().split()
                    fea_str.add(' '.join(tmpline[2:]))
                    new_tmpline = [float(tmpline[0])] + [int(t) for t in tmpline[1:]]
                    click_num += new_tmpline[1]
                    fea_l.update(new_tmpline[2:])
                line = fin.readline()
    print(num)
    print(doc_num)
    print("con_num:{}\tclick_num:{}".format(conversion_num, click_num))
    print(len(fea_str))
    print(len(fea_l), max(fea_l), min(fea_l))
    for k, v in occur.items():
        print(k, v)


def read_test(inf):
    def _parse_fn(record):
        features = {
            "click_y": tf.FixedLenFeature([], tf.float32),
            "conversion_y": tf.FixedLenFeature([], tf.float32),
            "features": tf.FixedLenFeature([10], tf.int64)
        }
        parsed = tf.parse_single_example(record, features)
        y = parsed.pop('click_y')
        z = parsed.pop('conversion_y')
        return parsed, {"click_y": y, "conversion_y": z}

    dataset = tf.data.TFRecordDataset(inf)
    dataset = dataset.map(_parse_fn)
    item = dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        for i in range(10):
            batch_features, batch_labels = sess.run(item.get_next())
            print(type(batch_features), type(batch_labels))
            print(batch_features, batch_labels)


if __name__ == '__main__':
    tr_files = glob.glob("Criteo/t*/*txt")
    print("train_files:", tr_files)
    read_test(tr_files[0])
