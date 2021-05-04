from sklearn.metrics import ndcg_score

pred = [[0.6, 0.5, 0.8, 0.1]]
label = [[1, 0, 0, 0]]
print(ndcg_score(y_true=label, y_score=pred))
