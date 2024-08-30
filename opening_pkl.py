import os
import pickle as cPickle
import pickle
import bz2

current_dir = os.getcwd()
f_pk1 = os.path.join(current_dir, 'outputs/test2-4.5cm/results.pkl')
print(f_pk1)

# load a pickle:
def load_pickle(filename):

    if filename[-4:] == 'pbz2':
        data = bz2.BZ2File(filename, 'rb')
        data = cPickle.load(data)
    else:
        dbfile = open(filename, 'rb')
        data = pickle.load(dbfile)
        dbfile.close()
    return data

pkl_file = load_pickle((f_pk1))
print(pkl_file)