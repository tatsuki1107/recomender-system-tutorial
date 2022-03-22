import argparse
import os
import numpy as np
from scipy.sparse import csc_matrix
from fastFM import als


def iter_feedbacks(filename):
    with open(filename) as f:
        for line in f:
            fields = line.strip().split('\t')
            yield ExplicitFeedback(*fields)


class ExplicitFeedback(object):

    def __init__(self, user_id, item_id, rating, timestamp):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating
        self.timstamp = timestamp


class Encoder(object):

    def __init__(self):
        self.id2index = {}

    def get_Xy(self, filename):
        rows, cols, data = [], [], []
        y = []
        for i, feedback in enumerate(iter_feedbacks(filename)):
            j = self.id2index.setdefault('user-' + feedback.user_id)
            rows.append(i)
            cols.append(j)
            data.append(1)
            j = self.id2index.setdefault('item-' + feedback.item_id)
            rows.append(i)
            cols.append(j)
            data.append(1)
            y.append(feedback.rating)
        X = csc_matrix((data, (rows, cols)), shape=(i, len(self.id2index)))
        y = np.array(y)
        return X, y


def main(args):
    encoder = Encoder()
    X, y = encoder.get_Xy(os.path.join(args.in_dir, 'ua.base'))
    fm = als.FMRegression(random_state=args.random_state)
    fm.fit(X, y)

    x_test, y_test = encoder.get_Xy(os.path.join(args.in_dir, 'ua.test'))
    y_pred = fm.predict(x_test)
    print(np.c_[y, y_pred][:10])


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    p.add_argument('--random-state', type=int, default=1)
    return p


if __name__ == "__main__":
    p = get_parser()
    args = p.parse_args()
    main(args)
