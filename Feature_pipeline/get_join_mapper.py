""" make train dateset
"""
import sys
from collections import defaultdict


def load_fcnts(if_str):
    feat_cnts_dict = defaultdict(lambda: None)
    with open(if_str) as f:
        for line in f:
            fid, new_id = line.rstrip().split()
            feat_cnts_dict[fid] = new_id
    return feat_cnts_dict


def doParseLog():
    """parse log on hdfs"""
    for line in sys.stdin:
        # recv = recv + 1
        try:
            splits = line.strip().split(',')
            split_len = len(splits)
            feat_lists = []
            # common_feature_index|feat_num|feat_list
            if split_len == 3:
                feat_strs = splits[2]
                for fstr in feat_strs.split('\x01'):
                    filed, feat_val = fstr.split('\x02')
                    feat, val = feat_val.split('\x03')
                    '''if filed == '101' or filed == '205' or feat_cnts_dict.get('{}:{}'.format(filed, feat)):
                        feat_lists.append('%s:%s:%s' % (filed, feat, val))'''
                    feat_lists.append('%s:%s:%s' % (filed, feat, val))
                # mapper把common_feature_index作为key，保证Skeleton 和 Common Features两份数据落到同一个reduce上
                print("{0}\t{1}\t{2}".format(splits[0], 'common', ' '.join(feat_lists)))
            # sample_id|y|z|common_feature_index|feat_num|feat_list
            elif split_len == 6:
                # y=0 & z=1过滤
                if splits[1] == '0' and splits[2] == '1':
                    continue
                feat_strs = splits[5]
                for fstr in feat_strs.split('\x01'):
                    filed, feat_val = fstr.split('\x02')
                    feat, val = feat_val.split('\x03')
                    '''if filed == '101' or filed == '205' or feat_cnts_dict.get('{}:{}'.format(filed, feat)):
                                            feat_lists.append('%s:%s:%s' % (filed, feat, val))'''
                    feat_lists.append('%s:%s:%s' % (filed, feat, val))

                # mapper把common_feature_index作为key，保证Skeleton 和 Common Features两份数据落到同一个reduce上
                print("{0}\t{1}\t{2} {3} {4} {5}".format(splits[3], 'sample', splits[0], splits[1], splits[2],
                                                         ' '.join(feat_lists)))
        except:
            continue
        # print recv


if __name__ == '__main__':
    '''feat_cnts_file, = sys.argv[1:2]
    feat_cnts_dict = load_fcnts(feat_cnts_file)'''
    doParseLog()
