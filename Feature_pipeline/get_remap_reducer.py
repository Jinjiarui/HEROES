import sys
from collections import defaultdict


def get_order(elem):
    return int(elem.split()[0])


def doParseLog():
    feat_id_dict = defaultdict(lambda: [])
    for line in sys.stdin:
        try:
            common, strs = line.rstrip().split('\t')
            feat_id_dict[common].append(strs)
        except:
            continue
    for k, v in feat_id_dict.items():
        v.sort(key=get_order)
        for _ in v:
            print('{0} {1}'.format(k, _))


if __name__ == '__main__':
    doParseLog()
