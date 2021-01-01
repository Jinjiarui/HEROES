""" re map feat_id """
import os
import re
import sys
from collections import defaultdict


def load_fcnts(if_str):
    feat_cnts_dict = defaultdict(lambda: None)
    user_id_dict = defaultdict(lambda: None)
    item_id_dict = defaultdict(lambda: None)
    with open(os.path.join(if_str, 'down_remap.txt')) as f:
        for line in f:
            fid, new_id = line.rstrip().split()
            feat_cnts_dict[fid] = new_id
    with open(os.path.join(if_str, 'down_user_remap.txt')) as f:
        for line in f:
            uid, new_id = line.rstrip().split()
            user_id_dict[uid] = new_id
    with open(os.path.join(if_str, 'down_item_remap.txt')) as f:
        for line in f:
            item_id, new_id = line.rstrip().split()
            item_id_dict[item_id] = new_id
    return feat_cnts_dict, user_id_dict, item_id_dict


def doParseLog(feat_cnts_dict, user_id_dict, item_id_dict):
    """parse log on hdfs"""
    for line in sys.stdin:
        try:
            line = line.rstrip()
            splits = re.split("[ ,]", line)
            user_id = 0
            item_id = 0
            # y=0 & z=1过滤
            if splits[2] == '0' and splits[3] == '1':
                continue
            # remap feat_id
            feat_lists = []
            for fstr in splits[4:]:
                f, fid, val = fstr.split(':')
                if f == '101':
                    user_id = int(user_id_dict.get(fid)) + 1
                elif f == '205':
                    item_id = int(item_id_dict.get(fid)) + 1
                new_id = feat_cnts_dict.get("{}:{}".format(f, fid))
                if new_id:
                    feat_lists.append('%s:%s:%s' % (f, new_id, val))
            print(
                "{0}\t{1} {2} {3} {4} {5} {6}".format(splits[0], splits[1], splits[2], splits[3], user_id, item_id,
                                                      ' '.join(feat_lists)))
        except:
            continue


if __name__ == '__main__':
    feat_cnts_file, = sys.argv[1:2]
    feat_cnts_dict, user_id_dict, item_id_dict = load_fcnts(feat_cnts_file)
    doParseLog(feat_cnts_dict, user_id_dict, item_id_dict)
