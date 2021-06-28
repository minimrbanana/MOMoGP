"""
Created on June 08, 2021
@author: Zhongjie Yu
@author: Mingye Zhu
"""
import numpy as np
from collections import Counter
import random


class Sum_:
    def __init__(self, **kwargs):
        self.maxs = kwargs['maxs']
        self.mins = kwargs['mins']
        self.dimension = dict.get(kwargs, 'dimension', None)
        self.children = dict.get(kwargs, 'children', [])
        self.depth = dict.get(kwargs, 'depth', 0)
        self.n = kwargs['n']
        self.scope = dict.get(kwargs, 'scope', [])
        self.parent = dict.get(kwargs, 'parent', None)
        self.splits = dict.get(kwargs, 'splits', [])
        self.idx = dict.get(kwargs, 'idx', [])
        self.pop = dict.get(kwargs, 'pop', [])
        self.y = dict.get(kwargs, 'y', [])


class Product_x_:
    def __init__(self, **kwargs):
        self.split = kwargs['split']
        self.splits = kwargs['splits']
        self.dimension = kwargs['dimension']
        self.depth = kwargs['depth']
        self.children = kwargs['children']
        self.parent = kwargs['parent']
        self.maxs = kwargs['maxs']
        self.mins = kwargs['mins']


class Product_y_:
    def __init__(self, **kwargs):
        self.children = kwargs['children']
        self.maxs = kwargs['maxsy']
        self.mins = kwargs['minsy']
        self.scope = dict.get(kwargs, 'scope', None)
        self.parent = dict.get(kwargs, 'parent', None)
        self.collect = dict.get(kwargs, 'collect', None)


class GPMixture:
    def __init__(self, **kwargs):
        self.mins = kwargs['mins']
        self.maxs = kwargs['maxs']
        self.idx = dict.get(kwargs, 'idx', [])
        self.parent = kwargs['parent']
        self.y = kwargs['y']


def _cached_gp(cache, **kwargs):
    min_, max_, y = list(kwargs['mins']), list(kwargs['maxs']), kwargs['y']
    cached = dict.get(cache, (*min_, *max_))
    if not cached:
        cache[(*min_, *max_)] = GPMixture(**kwargs)

    return cache[(*min_, *max_)]


def query(X, mins, maxs, skipleft=False):
    mask, D = np.full(len(X), True), X.shape[1]
    for d_ in range(D):
        if skipleft:
            mask = mask & (X[:, d_] >= mins[d_]) & (X[:, d_] < maxs[d_])
        else:
            ## consider data at split points to either side or both sides according to the situation
            mask = mask & (X[:, d_] >= mins[d_]) & (X[:, d_] <= maxs[d_])
    return np.nonzero(mask)[0]


def build_MOMoGP(**kwargs):
    ## bulid the MOMoGP structure
    X = kwargs['X']
    Y = kwargs['Y']
    ddd = dict.get(kwargs, 'qd', 0)
    min_idx = dict.get(kwargs, 'max_samples', 1)

    root_mixture_opts = {
        'mins': np.min(X, 0),
        'maxs': np.max(X, 0),
        'n': len(X),
        'scope': [i for i in range(Y.shape[1])],
        'parent': None,
        'dimension': np.argsort(-np.var(X, axis=0))[0],
        'idx': X,
        'y':Y
    }

    nsplits = Counter()
    root_node = Sum_(**root_mixture_opts)
    to_process, cache = [root_node], dict()
    count = 0
    while len(to_process):
        node = to_process.pop()
        if type(node) is Product_y_:
            for i in range(len(node.children)):
                node2 = node.children[i]

                if type(node2) is Sum_ :
                    d = node2.dimension
                    x_node = node2.idx
                    scope2 = node2.scope
                    y2 = node2.y
                    mins_node, maxs_node = np.min(x_node, 0), np.max(x_node, 0)
                    scope = node2.scope
                    d_selected = np.argsort(-np.var(x_node, axis=0))
                    d2 = d_selected[1]
                    sum_children = [1, 2]
                    ## take quantiles as split points
                    quantiles = np.quantile(x_node, np.linspace(0, 1, num = ddd+2), axis=0).T
                    # d = [d, d2]
                    d = [d_selected[0],d_selected[1]]
                    m = 0
                    for split in sum_children:
                        u = np.unique(quantiles[d[m]])  ##ignore duplicated split points

                        ## put data into different split intervals
                        loop = []
                        if len(u) == 1:
                            loop.append(x_node)
                        for i in range(len(u)-1):
                            new_maxs, new_mins = maxs_node.copy(), mins_node.copy()
                            skipleft = True
                            if i == 0:
                                skipleft = False
                            new_mins[d[m]] = u[i]
                            new_maxs[d[m]] = u[i + 1]
                            idx_i = query(x_node, new_mins, new_maxs, skipleft=skipleft)
                            if len(idx_i)==0:
                                print("empty children due to data")
                                continue
                            loop.append(idx_i)

                        next_depth = node2.depth + 1
                        results = []
                        ##create next-layer nodes for current node
                        for idx in loop:
                            x_idx = x_node[idx]
                            maxs_loop = np.max(x_idx, axis=0)
                            mins_loop = np.min(x_idx, axis=0)

                            # y_idx = y2[idx]
                            next_dimension = np.argsort(-np.var(x_idx, axis=0))[0]
                            if len(scope) == 1:
                                if len(idx) < min_idx and len(idx)>0:
                                    gp = []
                                    prod_opts = {
                                        'minsy': mins_loop,
                                        'maxsy': maxs_loop,
                                        'scope': scope,
                                        'children': gp,
                                    }
                                    prod = Product_y_(**prod_opts)
                                    a = _cached_gp(cache, mins=mins_loop, maxs=maxs_loop, idx=idx, y=scope[0], parent=None)
                                    gp.append(a)
                                    results.append(prod)
                                else:
                                    mixture_opts = {
                                        'mins': mins_loop,
                                        'maxs': maxs_loop,
                                        'depth': next_depth,
                                        'dimension': next_dimension,
                                        'n': len(idx),
                                        'scope': scope,
                                        'idx': x_idx,


                                    }
                                    results.append(Sum_(**mixture_opts))
                            else:
                                a = int(len(scope) / 2)
                                scope1 = random.sample(scope, a)
                                scope2 = list(set(scope) - set(scope1))
                                if len(idx) >= min_idx:
                                    mixture_opts1 = {
                                        'mins': mins_loop,
                                        'maxs': maxs_loop,
                                        'depth': next_depth,
                                        'dimension': next_dimension,
                                        'n': len(idx),
                                        'scope': scope1,
                                        'idx': x_idx,
                                    }
                                    mixture_opts2 = {
                                        'mins': mins_loop,
                                        'maxs': maxs_loop,
                                        'depth': next_depth,
                                        'dimension': next_dimension,
                                        'n': len(idx),
                                        'scope': scope2,
                                        'idx': x_idx,
                                    }
                                    prod_opts = {
                                        'minsy': mins_loop,
                                        'maxsy': maxs_loop,
                                        'scope': scope1 + scope2,
                                        'children': [Sum_(**mixture_opts1), Sum_(**mixture_opts2)]
                                    }

                                    prod = Product_y_(**prod_opts)
                                    results.append(prod)
                                else:
                                    gp = []
                                    prod_opts = {
                                        'minsy': mins_loop,
                                        'maxsy': maxs_loop,
                                        'scope': scope1+scope2,
                                        'children': gp,
                                    }

                                    prod = Product_y_(**prod_opts)
                                    for yi in prod.scope:
                                        a = _cached_gp(cache, mins=mins_loop, maxs=maxs_loop, idx=idx, y=yi, parent=None)
                                        gp.append(a)
                                        count += 1
                                    results.append(prod)

                        if len(results) != 1:
                            to_process.extend(results)
                            separator_opts = {
                                'depth': node2.depth,
                                'mins': mins_node,
                                'maxs': maxs_node,
                                'dimension': d[m],
                                'split': split,
                                'children': results,
                                'parent': None,
                                'splits':u
                            }
                            node2.children.append(Product_x_(**separator_opts))
                        elif len(results) == 1:
                            node2.children.extend(results)
                            to_process.extend(results)
                        else:
                            raise Exception('1')
                        m += 1

        elif type(node) is Sum_:
            d = node.dimension
            x_node = node.idx
            mins_node, maxs_node = np.min(x_node, 0), np.max(x_node, 0)
            scope = node.scope
            d_selected = np.argsort(-np.var(x_node, axis=0))
            d2 = d_selected[1]
            quantiles = np.quantile(x_node, np.linspace(0, 1, num = ddd+2),axis=0).T
            sum_children=[1,2]
            # d = [d, d2]
            d = [d_selected[0], d_selected[1]]
            m = 0
            # y = node.y
            for split in sum_children:
                u = np.unique(quantiles[d[m]])

                loop = []
                if len(u) == 1:
                    loop.append(x_node)

                for i in range(len(u)-1):
                    new_maxs, new_mins = maxs_node.copy(), mins_node.copy()
                    skipleft = True
                    if i == 0:
                        skipleft = False

                    new_mins[d[m]] = u[i]
                    new_maxs[d[m]] = u[i + 1]
                    idx_i = query(x_node, new_mins, new_maxs, skipleft=skipleft)
                    if len(idx_i)==0:
                        print("empty children due to data")
                        continue
                    loop.append(idx_i)

                next_depth = node.depth + 1
                results = []
                for idx in loop:
                    x_idx = x_node[idx]
                    maxs_loop = np.max(x_idx,axis=0)
                    mins_loop = np.min(x_idx,axis=0)
                    # y_idx = y[idx]
                    next_dimension = np.argsort(-np.var(x_idx, axis=0))[0]
                    if len(scope) == 1:
                        if len(idx) < min_idx and len(idx) >0:
                            gp = []
                            prod_opts = {
                                'minsy': mins_loop,
                                'maxsy': maxs_loop,
                                'scope': scope,
                                'children': gp,
                            }

                            prod = Product_y_(**prod_opts)
                            a = _cached_gp(cache, mins=mins_loop, maxs=maxs_loop, idx=idx, y=scope[0], parent=None)
                            gp.append(a)
                            results.append(prod)
                        else:
                            mixture_opts = {
                                'mins': mins_loop,
                                'maxs': maxs_loop,
                                'depth': next_depth,
                                'dimension': next_dimension,
                                'n': len(idx),
                                'scope': scope,
                                'idx': x_idx,
                                # 'y': y_idx
                            }
                            results.append(Sum_(**mixture_opts))

                    else:
                        a = int(len(scope) / 2)
                        scope1 = random.sample(scope, a)
                        scope2 = list(set(scope) - set(scope1))
                        if len(idx) >= min_idx:
                            mixture_opts1 = {
                                'mins': mins_loop,
                                'maxs': maxs_loop,
                                'depth': next_depth,
                                'dimension': next_dimension,
                                'n': len(idx),
                                'scope': scope1,
                                'idx': x_idx,
                                # 'y': y_idx
                            }
                            mixture_opts2 = {
                                'mins': mins_loop,
                                'maxs': maxs_loop,
                                'depth': next_depth,
                                'dimension': next_dimension,
                                'n': len(idx),
                                'scope': scope2,
                                'idx': x_idx,
                                # 'y': y_idx
                            }
                            prod_opts = {
                                'minsy': mins_loop,
                                'maxsy': maxs_loop,
                                'scope': scope1+scope2,
                                'children': [Sum_(**mixture_opts1),Sum_(**mixture_opts2)]
                            }

                            prod = Product_y_(**prod_opts)
                            results.append(prod)
                        else:
                            gp = []
                            prod_opts = {
                                'minsy': mins_loop,
                                'maxsy': maxs_loop,
                                'scope': scope1+scope2,
                                'children': gp,
                            }

                            prod = Product_y_(**prod_opts)
                            for yi in prod.scope:
                                a = _cached_gp(cache, mins=mins_loop, maxs=maxs_loop, idx=idx, y=yi, parent=None)
                                gp.append(a)
                                count+=1
                            results.append(prod)


                if len(results) != 1:
                    to_process.extend(results)
                    separator_opts = {
                        'depth': node.depth,
                        'mins': mins_node,
                        'maxs': maxs_node,
                        'dimension': d[m],
                        'split': split,
                        'children': results,
                        'parent': None,
                        'splits':u
                    }
                    node.children.append(Product_x_(**separator_opts))
                elif len(results) == 1:
                    node.children.extend(results)
                    to_process.extend(results)
                else:
                    raise Exception('1')
                m += 1

    gps = list(cache.values())
    leaf_len = [len(gp.idx) for gp in gps]

    print(f"Leaf observations:\t {leaf_len}\nSum:\t\t\t {sum(leaf_len)} (N={len(X)})")

    return root_node, gps
