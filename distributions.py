import numpy as np
import scipy.special

def chol_update(C, x):
    "Updates Cholesky matrix factorisation as per https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update"
    n = len(x)
    for idx in range(n):
        r = sqrt(C[idx, idx] ** 2 + x[idx] ** 2)
        c = r / C[idx, idx]
        s = x[idx] / C[idx, idx]
        C[idx, idx] = r
        C[(idx + 1):(n + 1), idx] = (C[(idx + 1):(n + 1), idx] + s*x[(idx + 1):(n + 1)])/c
        x[(idx + 1):(n + 1)] = c*x[(idx + 1):(n + 1)] - s*C[(idx + 1):(n + 1), idx]
    return C

class Distribution(object):
    """ Superclass to specify required functions and return messages if they are not yet implemented in subclass.
    """
    def __init__(self, data):
        self.data = data

    def validate_prior(self):
        "Ensures that required parameters are given in dictionary format"
        print('Parameter validation code needs to go here.')
        pass

    def log_pred(self):
        "Returns log predictive probability - probability of a data point belonging to distribution given other data"
        print('Log predictive code needs to go here.')
        pass

    def log_marg(self):
        "Returns log marginal probability"
        print('Log marginal code needs to go here.')
        pass

    def add_data(self):
        "Adds a data point to the component"
        print('Adding data code needs to go here.')
        pass

    def rem_data(self):
        "Removes a data point from the component"
        print('Removing data code needs to go here.')
        pass   

    def log_like(self):
        "Returns log likelihood"
        print('Log likelihood code needs to go here.')
        pass

class Gaussian(Distribution):

    def __init__(self, data, prior):
        super(Gaussian, self).__init__(data)
        self.prior = prior
        if self.validate_prior(prior):
            print('Prior is suitable.')
        else:
            print('Prior is not suitable.')

        self.params = dict()
        self.params['dimensions'] = self.prior['d']
        self.params['rel_variance'] = self.prior['r']
        self.params['dof'] = self.prior['v']
        self.params['member_count'] = 0
        self.params['cholesky'] = np.linalg.cholesky(self.prior['S'] + self.prior['r']*self.prior['m']*self.prior['m'].T)
        self.params['member_sum'] = self.prior['r']*self.prior['m']
        self.params['init_log_likelihood'] = self.log_like(self)

    def validate_prior(self, prior):
        "Ensures that a dictonary representing a Normal-Wishart prior has been passed."
        if not isinstance(prior, dict):
            print('Provided prior is not a dictionary.')
            return False

        key_list = prior.keys()
        required_keys = ['d', 'r', 'v', 'm', 'S']

        for k in required_keys:
            if k not in key_list:
                print('Prior parameter ', k, ' not provided.')
                return False 

        if prior['d'] != self.data.shape[1]:
            print('Prior dimension does not match data.')
            return False

        if not isinstance(prior['r'], int):
            print('Prior relative precision is not an integer.')
            return False

        if not isinstance(prior['v'], int):
            print('Prior degrees of freedom is not an integer.')
            return False    

        if prior['S'].size != self.data.shape[1] ** 2: 
            print('Prior precision does not match data shape.')
            return False

        return True

    def add_data(self, data_point):
        "Assigns data_point to this Gaussian and updates significant statistics as required."
        self.params['member_count'] += 1
        self.params['rel_variance'] += 1
        self.params['dof'] += 1
        self.params['cholesky'] = chol_update(self.params['cholesky'], data_point, '+')
        self.params['member_sum'] += data_point

    def rem_data(self, data_point):
        "Removes data_point from this Gaussian and updates significant statistics as required."
        self.params['member_count'] -= 1
        self.params['rel_variance'] -= 1
        self.params['dof'] -= 1
        self.params['cholesky'] = chol_update(self.params['cholesky'], data_point, '-')
        self.params['member_sum'] -= data_point

    def log_marg(self):
        "Computes log marginal for this Gaussian."
        return self.log_like(self) - self.params['init_log_likelihood']

    def log_pred(self, data_point):
        "Computes log predictive for this Gaussian and given data point."
        log_like_tmp = self.log_like()
        self.add_data(data_point)
        log_like_new = self.log_like()
        self.rem_data(data_point)
        return log_like_new - log_like_tmp

    def log_like(self):
        "Computes log likelihood for this Gaussian."

        n = self.params['member_count']
        d = self.params['dimensions']
        r = self.params['rel_variance']
        C = self.params['cholesky']
        X = self.params['member_sum']
        v = self.params['dof']

        log_likelihood = -n*d/2*np.log(np.pi) - d/2*np.log(r) - v*np.sum(np.log(np.diag(chol_update(C, X/np.sqrt(r), '-'))) + np.sum(scipy.special.gammaln([(v - x)/2. for x in range(0, d)]))

        return log_likelihood

def test(x, y):
    print('In test fn')
    return x + y