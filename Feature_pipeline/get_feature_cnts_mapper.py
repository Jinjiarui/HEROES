import sys
from collections import defaultdict


def doParseLog():
    feat_id_dict = defaultdict(lambda: 0)
    for line in sys.stdin:
        # recv = recv + 1
        try:
            splits = line.strip().split()
            '''split_len = len(splits)
            # common_feature_index|feat_num|feat_list
            if split_len == 3:
                feat_strs = splits[2]
                for fstr in feat_strs.split('\x01'):
                    filed, feat_val = fstr.split('\x02')
                    feat, val = feat_val.split('\x03')
                    feat_id_dict["{}:{}".format(filed, feat)] += 1
            # sample_id|y|z|common_feature_index|feat_num|feat_list
            elif split_len == 6:
                # y=0 & z=1过滤
                if splits[1] == '0' and splits[2] == '1':
                    continue
                feat_strs = splits[5]
                for fstr in feat_strs.split('\x01'):
                    filed, feat_val = fstr.split('\x02')
                    feat, val = feat_val.split('\x03')
                    feat_id_dict["{}:{}".format(filed, feat)] += 1'''
            for fstr in splits[4:]:
                filed, feat, val = fstr.split(':')
                feat_id_dict["{}:{}".format(filed, feat)] += 1
        except:
            continue
    for feat_id, cnts in feat_id_dict.items():
        print("{0}\t{1}".format(feat_id, cnts))


if __name__ == '__main__':
    doParseLog()
