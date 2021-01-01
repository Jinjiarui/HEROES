import sys

from collections import defaultdict


def doParseLog():
    feat_id_dict = defaultdict(lambda: 0)
    for line in sys.stdin:
        try:
            feat_id, cnts = line.rstrip().split('\t')
            feat_id_dict[feat_id] += int(cnts)
        except:
            continue
    for feat_id, cnts in feat_id_dict.items():
        print("{0}\t{1}".format(feat_id, cnts))


if __name__ == '__main__':
    doParseLog()
