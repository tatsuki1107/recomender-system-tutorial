import argparse


class ExplicitFeedback(object):

    def __init__(self, user_id, item_id, rating, timestamp):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating
        self.timstamp = timestamp


def item_feedbacks(filename):
    with open(filename) as f:
        for line in f:
            fields = line.strip().split('\t')
            yield ExplicitFeedback(*fields)


def main(args):
    pass


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', required=True)
    return p


if __name__ == "__main__":
    p = get_parser()
    args = p.parse_args()
    main(args)
