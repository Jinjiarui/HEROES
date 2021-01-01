import numpy as np
from scipy import sparse


def loadCriteo(file, feature_loc, label_loc):
    with open(file, 'r') as fin:
        l = fin.readline()
        user_dir = []
        feature_dir = {}
        while l:
            (sql_len, label) = [int(_) for _ in l.split(' ')]
            user_dir.append(set())
            for _ in range(sql_len):
                tmp_line = fin.readline().split(' ')
                click = int(tmp_line[1])
                feature = ' '.join(tmp_line[2:])
                if feature not in feature_dir.keys():
                    feature_dir[feature] = len(feature_dir)
                user_dir[-1].add(feature_dir[feature])
            l = fin.readline()
        result = sparse.lil_matrix((len(user_dir), len(feature_dir)))
        for i in range(len(user_dir)):
            for j in user_dir[i]:
                result[i, j] = 1
        result = result.tocsc()
        sparse.save_npz('Criteo/result.npz', result)


if __name__ == '__main__':
    loadCriteo('Criteo/test_usr.yzx.txt', 1, 2)
