import numpy as np
import csv
import sys
import os
import h5py
import pandas as pd
import simplejson as json
import sqlite3
import copy

# structure followed in this file is based on : https://github.com/nhammerla/deepHAR/tree/master/data
# and https://github.com/IRC-SPHERE/sphere-challenge

class data_reader:
    def __init__(self, dataset):
        if dataset =='opp':
            self.data, self.idToLabel = self.readOpportunity()
            self.save_data_csv(dataset)
            self.save_data(dataset)
        else:
            print('Not supported yet')
            sys.exit(0)

    def save_data_csv(self, dataset):
        if dataset == 'opp':
            for key in self.data:
                for field in self.data[key]:
                    np.savetxt('data_replace_'+field+'.csv', self.data[key][field], delimiter=',')
            print('Saved in csv.')
        else:
            print('Not supported yet')
            sys.exit(0)

    def save_data(self, dataset):
        if dataset == 'opp':
            f = h5py.File('opportunity_replace.h5')
            for key in self.data:
                f.create_group(key)
                for field in self.data[key]:
                    f[key].create_dataset(field, data=self.data[key][field])
            f.close()
            print('Done.')
            #np.savetxt("test.csv", f, '%g', ',')
        else:
            print('Not supported yet')
            sys.exit(0)


    @property
    def train(self):
        return self.data['train']

    @property
    def test(self):
        return self.data['test']

    def readOpportunity(self):
        files = {
            'train': ['S1-ADL1.dat','S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat', 'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL5.dat', 'S2-Drill.dat', 'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL5.dat', 'S3-Drill.dat', 'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'],
            'test': ['S2-ADL3.dat', 'S2-ADL4.dat','S3-ADL3.dat', 'S3-ADL4.dat']
        }
        #names are from label_legend.txt of Opportunity dataset
        #except 0-ie Other, which is an additional label
        label_map = [
            (0,      'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        cols = [
            37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 56, 57, 58,63, 64, 65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 80, 81, 82, 83, 84,
            89, 90, 91, 92, 93, 94, 95, 96, 97, 102, 103, 104, 105, 106, 107, 108,109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 249
            ]

        data = {dataset: self.readOpportunityFilesRaplace(files[dataset], cols, labelToId)
                for dataset in ('train', 'test')}

        return data, idToLabel

#this is from https://github.com/nhammerla/deepHAR/tree/master/data and it is an opportunity Challenge reader. It is a python translation one
#for the official one provided by the dataset publishers in Matlab.
    def readOpportunityFilesRaplace(self, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i+1, len(filelist)))
            with open('/Users/JinWei/Downloads/OpportunityUCIDataset/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                pre_line = next(reader)
                pre_line = [0 if x == 'NaN' else x for x in pre_line]
                while(True):
                    try:
                        cur_line = next(reader)
                        elem = []
                        for ind in cols:
                            if cur_line[ind] == 'NaN':
                                cur_line[ind] = pre_line[ind]
                            elem.append(cur_line[ind])
                        pre_line = cur_line
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

                        if 'NaN' in elem:
                            print(elem)
                    except StopIteration:
                        break
        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1}

if __name__ == "__main__":
    print('Reading opportunity dataset')
    dr = data_reader('opp')
