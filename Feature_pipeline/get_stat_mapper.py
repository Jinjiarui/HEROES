import sys
from collections import defaultdict


def doParseLog():
    """parse log on hdfs"""
    cnts_dict = defaultdict(lambda: 0)
    for line in sys.stdin:
        # recv = recv + 1
        try:
            splits = line.strip().split(',')
            for fstr in splits[3].split(' '):
                feat, _ = fstr.rsplit(':', 1)
                cnts_dict[feat] += 1
        except:
            continue
    for key, val in cnts_dict.items():
        print("{0}\t{1}".format(key, val))


if __name__ == '__main__':
    doParseLog()
