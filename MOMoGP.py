"""
Created on June 08, 2021
@author: Zhongjie Yu
@author: Mingye Zhu
"""
import numpy as np
from MOMoGPstructure import Sum_, Product_x_, GPMixture, Product_y_
import gc
import torch
import gpytorch
from gpytorch.kernels import *
from gpytorch.likelihoods import *
from gpytorch.mlls import *
from torch.optim import *
from scipy.special import logsumexp


class Sum:
    def __init__(self, **kwargs):
        self.children = []
        self.weights = []
        self.scope=kwargs['scope']
        self.mll = []
        return None
    
    def forward(self, x_pred, **kwargs):
        if len(self.children) == 1:
            r_ = self.children[0].forward(x_pred, **kwargs)
            mu_x = r_[0]
            co_x = r_[1]
            #return r_[0],r_[1]
        elif len(self.children) == 2:
            _wei = np.array(self.weights).reshape((-1,1))
            c,a= self.children[0].forward(x_pred, **kwargs)
            d,b = self.children[1].forward(x_pred, **kwargs)
            mu_x = c*self.weights[0]+d*self.weights[1]
            co1=a*self.weights[0]+b*self.weights[1]
            t3 = np.matmul(c,c.transpose((0,2,1)))
            t6 = np.matmul(d,d.transpose((0,2,1)))
            co2 = t3*self.weights[0]+t6*self.weights[1]
            e = c*self.weights[0]+d*self.weights[1]
            co3 = np.matmul(e,e.transpose((0,2,1)))
            co_x = co1+co2-co3 # compute the covariance matrix of the mixture distribution

        elif len(self.children)==4:
            _wei = np.array(self.weights).reshape((-1, 1))
            c, a = self.children[0].forward(x_pred, **kwargs)
            d, b = self.children[1].forward(x_pred, **kwargs)
            m3, c3 = self.children[2].forward(x_pred, **kwargs)
            m4, c4 = self.children[3].forward(x_pred, **kwargs)
            mu_x = c * self.weights[0] + d * self.weights[1]+m3*self.weights[2]+m4*self.weights[3]
            co1 = a * self.weights[0] + b * self.weights[1]+c3*self.weights[2]+c4*self.weights[3]
            t3 = np.matmul(c, c.transpose((0, 2, 1)))
            t6 = np.matmul(d, d.transpose((0, 2, 1)))
            t4 = np.matmul(m3, m3.transpose((0, 2, 1)))
            t7 = np.matmul(m4, m4.transpose((0, 2, 1)))
            co2 = t3 * self.weights[0] + t6 * self.weights[1]+t4*self.weights[2]+t7*self.weights[3]
            e = c * self.weights[0] + d * self.weights[1]+m3*self.weights[2]+m4*self.weights[3]
            co3 = np.matmul(e, e.transpose((0, 2, 1)))
            co_x = co1 + co2 - co3  # compute the covariance matrix of the mixture distribution

        else:
            raise NotImplementedError

        return mu_x, co_x

    def update(self):
        c_mllh = np.array([c.update() for c in self.children])
        lw = logsumexp(c_mllh)
        logweights = c_mllh - lw
        ## add a prior in case one child has too large a weight
        self.weights = np.exp(logweights) + np.ones(len(logweights)) * 0.5
        self.weights = self.weights / np.sum(self.weights)

        return lw

    def update_mll(self):
        if len(self.children)==2:
            a = torch.stack((torch.tensor(self.children[0].update_mll()),torch.tensor(self.children[1].update_mll())))
            w_log = torch.logsumexp(a,0)+torch.tensor(0.5)
            
            return w_log
        else:
            raise NotImplementedError


class Product_x:
    def __init__(self, **kwargs):
        self.children = []
        self.split = kwargs['split']
        self.dimension = kwargs['dimension']
        self.depth = kwargs['depth']
        self.splits = kwargs['splits']
        return None

    def forward(self, x_pred, **kwargs):
        y_d = dict.get(kwargs, 'y_d', 0)
        mu_x = np.zeros((len(x_pred),1, y_d))
        co_x = np.zeros((len(x_pred), y_d,y_d))
        idx_all = []
        for i, child in enumerate(self.children):
            if i < len(self.children) :
                if i == 0:
                    idx = np.where((x_pred[:, self.dimension] <= self.splits[i + 1]))[0]
                elif i == len(self.children)-1:
                    idx = np.where((x_pred[:, self.dimension] > self.splits[i]))[0]
                else:

                    idx = np.where((x_pred[:, self.dimension]>self.splits[i]) & (x_pred[:, self.dimension] <= self.splits[i+1]))[0]
                idx_all.append(idx)
                mu_x[idx, :, :], co_x[idx, :, :] = child.forward(x_pred[idx], **kwargs)

        return mu_x, co_x

    def update(self):
        return np.sum([c.update() for c in self.children])

    def update_mll(self):
        sum = 0
        for c in self.children:
            sum+=c.update_mll()

        return sum


class Product_y:
    def __init__(self, **kwargs):
        self.children = []
        self.scope = kwargs['scope']

        return None

    def forward(self, x_pred, **kwargs):
        y_d = dict.get(kwargs, 'y_d', 0)
        mu_x = np.zeros((len(x_pred),1, y_d))
        co_x = np.zeros((len(x_pred), y_d,y_d ))

        if type(self.children[0]) is Sum:
            for i, child in enumerate(self.children):
                mu_c, co_c= child.forward(x_pred, **kwargs)
                mu_x += mu_c
                co_x += co_c
            return mu_x, co_x
        elif type(self.children[0]) is GP:
            for i, child in enumerate(self.children):
                mu_c, co_c= child.forward(x_pred, **kwargs)
                mu_x[:,0, child.scope], co_x[:, child.scope, child.scope] = mu_c.squeeze(-1), co_c.squeeze(-1)
            return mu_x, co_x

    def update(self):
        return np.sum([c.update() for c in self.children])

    def update_mll(self):
        sum = 0
        for c in self.children:
            sum+=c.update_mll()
        return sum


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, **kwargs):
        x = dict.get(kwargs,'x')
        y = dict.get(kwargs,'y')
        likelihood = dict.get(kwargs,'likelihood')
        gp_type = dict.get(kwargs,'type')
        xd = x.shape[1]
        active_dims = torch.tensor(list(range(xd)))

        super(ExactGPModel, self).__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if gp_type == 'mixed':
            m = MaternKernel(nu=1.5, ard_num_dims=xd, active_dims=active_dims)
            l = PeriodicKernel()
            self.covar_module = ScaleKernel(m + l)
            return
        elif gp_type == 'matern0.5':
            k = MaternKernel(nu=0.5)
        elif gp_type == 'matern1.5':
            k = MaternKernel(nu=1.5)
        elif gp_type == 'matern2.5':
            k = MaternKernel(nu=2.5)
        elif gp_type == 'rbf':
            k = RBFKernel()
        elif gp_type == 'matern0.5_ard':
            k = MaternKernel(nu=0.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'matern1.5_ard':
            k = MaternKernel(nu=1.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'matern2.5_ard':
            k = MaternKernel(nu=2.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'rbf_ard':
            k = RBFKernel(ard_num_dims=xd)
        elif gp_type == 'linear':
            k = LinearKernel(ard_num_dims=xd)  # ard_num_dims for linear doesn't actually work
        else:
            raise Exception("Unknown GP type")

        self.covar_module = ScaleKernel(k)
        lengthscale_prior = gpytorch.priors.GammaPrior(2, 3)
        outputscale_prior = gpytorch.priors.GammaPrior(2, 3)
        self.covar_module.base_kernel.lengthscale = lengthscale_prior.sample()
        self.covar_module.outputscale = outputscale_prior.sample()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP:
    def __init__(self, **kwargs):
        self.type = kwargs['type']
        self.mins = kwargs['mins']
        self.maxs = kwargs['maxs']
        self.mll = 0
        self.x = dict.get(kwargs, 'x', [])
        self.scope = kwargs['scope']
        self.n = None
        self.y = dict.get(kwargs, 'y', [])
        self.count = kwargs['count']

    def forward(self, X_s, **kwargs):

        device_ = torch.device("cuda" if self.cuda else "cpu")
        self.model = self.model.to(device_)
        self.model.eval()
        self.likelihood.eval()
        x = torch.from_numpy(X_s).float().to(device_)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(x))
                pm, pv = observed_pred.mean, observed_pred.variance
                pm_ = pm.detach().cpu()
                pv_ = pv.detach().cpu()
                del observed_pred,pm,pv
                del self.model
                torch.cuda.empty_cache()
                gc.collect()
        x.detach()
        del x
        torch.cuda.empty_cache()
        gc.collect()

        return pm_, pv_

    def update(self):
        return self.mll

    def update_mll(self):
        return self.mll_grad

    def init(self, **kwargs):
        lr = dict.get(kwargs, 'lr', 0.2)
        steps = dict.get(kwargs, 'steps', 100)
        iter = dict.get(kwargs, 'iter', 0)
        self.n = len(self.x)
        self.cuda = dict.get(kwargs, 'cuda') and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        if self.cuda:
            torch.cuda.empty_cache()

        self.x = torch.from_numpy(self.x).float().to(self.device)
        self.y = torch.from_numpy(self.y.ravel()).float().to(self.device)

        self.likelihood = GaussianLikelihood()
        self.likelihood.train()
        self.model = ExactGPModel(x=self.x, y=self.y, likelihood=self.likelihood, type='matern1.5_ard').to(
        self.device)  # .cuda()
        self.optimizer = Adam([{'params': self.model.parameters()}], lr=lr)
        self.model.train()
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        print(f"\tGP {self.type} init completed. Training on {self.device}")
        iner_LMM = np.zeros((1,steps))
        for i in range(steps):

            self.optimizer.zero_grad()  # Zero gradients from previous iteration
            output = self.model(self.x)  # Output from model
            loss = -mll(output, self.y)  # Calc loss and backprop gradients
            loss.backward()
            self.optimizer.step()
            iner_LMM[:, i] = loss.detach().item()

        # LOG LIKELIHOOD NOW POSITIVE
        self.mll = -loss.detach().item()
        self.loss = loss.detach().item()

        self.x.detach()
        self.y.detach()

        del self.x
        del self.y
        self.x = self.y = None

        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
        return iner_LMM


def structure(root_region, scope,**kwargs):
    count=0
    root = Sum(scope=scope)
    to_process, gps = [(root_region, root)], dict()
    gp_types = dict.get(kwargs, 'gp_types', ['matern1.5_ard'])

    while len(to_process):
        gro, sto = to_process.pop()
        # sto = structure object
        if type(gro) is Sum_:
            for child in gro.children:
                if type(child) is Product_x_:
                    _child = Product_x(split=child.split, depth=child.depth, dimension=child.dimension,splits=child.splits)
                    sto.children.append(_child)
                    _cn = len(sto.children)
                    sto.weights = np.ones(_cn) / _cn
                elif type(child) is Product_y_:
                    scope = child.scope
                    _child = Product_y(scope = scope)
                    sto.children.append(_child)
                else:
                    print(type(child))
                    raise Exception('1')
            to_process.extend(zip(gro.children, sto.children))
        elif type(gro) is Product_x_: # sto is Product_x
            for child in gro.children:
                if type(child) is Product_y_:
                    scope = child.scope
                    _child = Product_y(scope = scope)
                    sto.children.append(_child)
                elif type(child) is Sum_:
                    scope = child.scope
                    _child = Sum(scope=scope)
                    sto.children.append(_child)
                elif type(child) is Product_x_:
                    _child = Product_x(split=child.split, depth=child.depth, dimension=child.dimension,splits=child.splits)
                    sto.children.append(_child)
                    _cn = len(sto.children)
                    sto.weights = np.ones(_cn) / _cn
                else:
                    raise Exception('1')

            to_process.extend(zip(gro.children, sto.children))

        elif type(gro) is Product_y_:
            i = 0
            for child in gro.children:
                if type(child) is Sum_:
                    scope = child.scope
                    _child = Sum(scope = scope)
                    sto.children.append(_child)
                elif type(child) is GPMixture:
                    gp_type = gp_types[0]
                    scopee = gro.scope

                    key = (*gro.mins, *gro.maxs, gp_type, count, scopee[i])
                    gps[key] = GP(type=gp_type, mins=gro.mins, maxs=gro.maxs, count=count,
                                  scope=scopee[i])
                    count+=1
                    sto.children.append(gps[key])
                    i += 1
                else:
                    raise Exception('1')

            to_process.extend(zip(gro.children, sto.children))

    return root, list(gps.values())
