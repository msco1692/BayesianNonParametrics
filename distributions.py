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

def test(x, y):
    print('In test fn')
    return x + y