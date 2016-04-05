import numpy as np
import scipy.special

def chol_update(C, x, sign = '+'):
    """Updates Cholesky matrix factorisation as per https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update. 
    Note that C must be a numpy array of floats and similarly for the row vector x. 
    Sign is a string containing + or - depending on if a Cholesky update or downdate is desired."""
    C_tmp = np.asarray([C.copy()])
    x_tmp = np.asarray([x.copy()])
    #n = len(x)
    n = 1
    for idx in range(n):
        if sign == '+':
            r = np.sqrt(C_tmp[idx] ** 2 + x_tmp[idx] ** 2)
        else:
            r = np.sqrt(C_tmp[idx] ** 2 - x_tmp[idx] ** 2)
        c = r / C_tmp[idx]
        s = x_tmp[idx] / C_tmp[idx]
        C_tmp[idx] = r
    return C_tmp[0]


    for idx in range(n):
        if sign == '+':
            r = np.sqrt(C_tmp[idx, idx] ** 2 + x_tmp[idx] ** 2)
        else:
            r = np.sqrt(C_tmp[idx, idx] ** 2 - x_tmp[idx] ** 2)
        c = r / C_tmp[idx, idx]
        s = x_tmp[idx] / C_tmp[idx, idx]
        C_tmp[idx, idx] = r
        if sign == '+':
            C_tmp[(idx + 1):(n + 1), idx] = (C_tmp[(idx + 1):(n + 1), idx] + s*x_tmp[(idx + 1):(n + 1)])/c
        else:
            C_tmp[(idx + 1):(n + 1), idx] = (C_tmp[(idx + 1):(n + 1), idx] - s*x_tmp[(idx + 1):(n + 1)])/c
        x_tmp[(idx + 1):(n + 1)] = c*x_tmp[(idx + 1):(n + 1)] - s*C_tmp[(idx + 1):(n + 1), idx]
    return C_tmp

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

class Gaussian(Distribution):
    """ Multivariate Gaussian class - subclass of Distribution. 

        Initialise with Gaussian(data, prior). prior specifies a dictionary as defined below, while data is a 2-dimensional numpy array with rows specifying data points.

        Attributes: 

        prior - dictionary specifying Normal-Wishart prior. prior must have the following entries:
                - 'd' - dimension of Gaussian
                - 'r' - relative precision 
                - 'v' - degrees of freedom of Wishart
                - 'm' - mean vector
                - 'S' - inverse scale matrix
        params - dictionary with posterior parameters and specific statistics for Gaussian distribution using the following entries:
                - 'dimensions' - dimension of Gaussian
                - 'rel_precision' - relative precision
                - 'dof' - degrees of freedom
                - 'member_count' - number of member data points
                - 'cholesky' - cholesky factorisation of mean precision 
                - 'member_sum' - sum of member data points
                - 'init_norm_constant' - log normalisation constant of prior

        Methods:

        validate_prior(self, prior) - Validates entries of prior dictionary as defined above
        add_data(self, data_point) - Assigns a data point to the Gaussian for Gibbs sampling
        rem_data(self, data_point) - Removes a data point from the Gaussian for Gibbs sampling
        log_marg(self) - Returns log marginal probability for the Gaussian
        log_pred(self, data_point) - Computes probability of a given data point belonging with the others in the Gaussian
        norm_constant(self) - Computes logarithm of normalisation constant for Gaussian with given data points
    """
    def __init__(self, data, prior):
        super(Gaussian, self).__init__(data)
        self.prior = prior
        # if self.validate_prior(prior):
        #     # print('Prior is suitable.')
        # else:
        #     # print('Prior is not suitable.')

        self.params = dict()
        self.params['dimensions'] = self.prior['d']
        self.params['rel_variance'] = self.prior['r']
        self.params['dof'] = self.prior['v']
        self.params['member_count'] = 0
        if self.params['dimensions'] == 1:
            self.params['cholesky'] = self.prior['S'] + self.prior['r']*self.prior['m']*self.prior['m'].T
        else:
            self.params['cholesky'] = np.linalg.cholesky(self.prior['S'] + self.prior['r']*self.prior['m']*self.prior['m'].T)
        self.params['member_sum'] = self.prior['r']*self.prior['m']
        self.params['init_norm_constant'] = self.norm_constant()

    def validate_prior(self, prior):
        "Ensures that a dictonary representing a Normal-Wishart prior has been passed."

        return True

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
        self.params['cholesky'] = chol_update(self.params['cholesky'], np.asarray(data_point), '+')
        self.params['member_sum'] += data_point

    def rem_data(self, data_point):
        "Removes data_point from this Gaussian and updates significant statistics as required."
        self.params['member_count'] -= 1
        self.params['rel_variance'] -= 1
        self.params['dof'] -= 1
        self.params['cholesky'] = chol_update(self.params['cholesky'], np.asarray(data_point), '-')
        self.params['member_sum'] -= data_point

    def log_marg(self):
        "Computes log marginal for this Gaussian."
        return self.norm_constant() - self.params['init_norm_constant']

    def log_pred(self, data_point):
        "Computes log predictive for this Gaussian and given data point."
        norm_constant_tmp = self.norm_constant()
        self.add_data(data_point)
        norm_constant_new = self.norm_constant()
        self.rem_data(data_point)
        return norm_constant_new - norm_constant_tmp

    def norm_constant(self):
        "Computes log likelihood for this Gaussian."
        n = self.params['member_count']
        d = self.params['dimensions']
        r = self.params['rel_variance']
        C = self.params['cholesky']
        X = self.params['member_sum']
        v = self.params['dof']
      
        if d == 1:
            norm_constant = -n*d/2*np.log(np.pi) - d/2*np.log(r) - v*np.sum(np.log(np.asarray(chol_update(C, np.asarray(X/np.sqrt(r)), '-')))) + np.sum(scipy.special.gammaln([(v - x)/2. for x in range(0, int(d))]))
        else:  
            norm_constant = -n*d/2*np.log(np.pi) - d/2*np.log(r) - v*np.sum(np.log(np.diag(np.asarray(chol_update(C, np.asarray(X/np.sqrt(r)), '-'))))) + np.sum(scipy.special.gammaln([(v - x)/2. for x in range(0, int(d))]))

        return norm_constant
