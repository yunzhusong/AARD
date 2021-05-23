import os
import pdb
import argparse


def main(args):

    train_path = os.path.join(args.data_path, 'split_0/train.label.txt')
    test_path = os.path.join(args.data_path, 'split_0/test.label.txt')
    event = args.leave_event

    with open(train_path) as f:
        train_data = f.read().split('\n')


    with open(test_path) as f:
        test_data = f.read().split('\n')

    all_data = train_data + test_data

    train_data, test_data = [], []
    for data in all_data:
        if event in data:
            test_data.append(data)
        else:
            train_data.append(data)

    print('Leaving {} event as test data'.format(event))
    print('Num of test data: {}'.format(len(test_data)))
    print('Num of train data: {}'.format(len(train_data)))

    
    out_path = os.path.join(args.data_path, 'split_{}'.format(event))
    os.makedirs(out_path, exist_ok=True)
    train_path = os.path.join(out_path, 'train.label.txt')
    test_path = os.path.join(out_path, 'test.label.txt')

    with open(train_path, 'w') as f:
        f.write('\n'.join(train_data))

    with open(test_path, 'w') as f:
        f.write('\n'.join(test_data))






if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-leave_event', type=str, default='charliehebdo')
    parser.add_argument('-data_path', type=str)
    args = parser.parse_args()
    main(args)



