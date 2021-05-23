import os
import pdb
import argparse
import numpy as np

import re

in_dir = './results/from_csv.txt'


def cal_average(args):
    with open(args.raw_txt) as f:
        raw = f.read()
        lines = raw.split('\n')


    alldata = []
    for line in lines:
        values = line.split()
        data = []
        for value in values:

            result = re.search(r'\d.\d+', value)
            if result:
                data.append(float(result.group()))
        if data:
            alldata.append(data)

    alldata = np.array(alldata)
    assert len(alldata)==5
    avg = np.average(alldata, 0)
    print('reading {}'.format(args.raw_txt))
    print(avg)

    #with open(args.raw_txt, 'a') as f:
    #    f.write('avg:\n{}'.format(avg))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-raw_txt', type=str)
    
    args = parser.parse_args()
    cal_average(args)


        






