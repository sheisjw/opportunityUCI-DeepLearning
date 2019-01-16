
    def readOpportunityFilesRaplace(self, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i+1, len(filelist)))
            with open('/Users/m193-hb/Downloads/OpportunityUCIDataset/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                pre_line = next(reader)
                firstrow = [0 if x == 'NaN' else x for x in pre_line]
                while(True):
                    try:
                        cur_line = next(reader)
                        elem = []
                        for ind in cols:
                            elem.append(cur_line[ind])
                            if cur_line[ind] == 'NaN':
                                cur_line[ind] = pre_line[ind]
                        pre_line = cur_line
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])
                    except:
                        break
        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1}