import numpy as np
import math
import copy


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def get_max_ndcg(k, *ins):
    l = [i for i in ins]
    l = copy.copy(l[0])
    l.sort(None,None,True)
    #print l
    max = 0.0
    for i in range(k):
        #print l[i]/math.log(i+2,2)
        max += (math.pow(2, l[i])-1)/math.log(i+2,2)
        #max += l[i]/math.log(i+2,2)
    return max

def get_ndcg(s, k):
    '''This is a function to get ndcg '''
    z = get_max_ndcg(k, s)
    dcg = 0.0
    for i in range(k):
        #print s[i]/math.log(i+2,2)
        dcg += (math.pow(2, s[i])-1)/math.log(i+2,2)
        #dcg += s[i]/math.log(i+2,2)
    if z ==0:
        z = 1;
    ndcg = dcg/z
    #print "Line:%s, NDCG@%d is %f with DCG = %f, z = %f"%(s, k, ndcg,dcg, z)
    return ndcg

def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max