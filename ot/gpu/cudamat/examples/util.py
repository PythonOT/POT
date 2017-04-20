from __future__ import division
import gzip
try: import cPickle as pickle
except: import pickle

def save(fname, var_list, source_dict):
    var_list = [var.strip() for var in var_list.split() if len(var.strip())>0]
    fo = gzip.GzipFile(fname, 'wb')
    pickle.dump(var_list, fo)
    for var in var_list:
        pickle.dump(source_dict[var], fo, protocol=2)
    fo.close()

def load(fname, target_dict, verbose = True):
    fo = gzip.GzipFile(fname, 'rb')
    var_list = pickle.load(fo)
    if verbose:
        print(var_list)
    for var in var_list:
        target_dict[var] = pickle.load(fo)
    fo.close()

