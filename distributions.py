class Distribution(object):
    """ Superclass to specify required functions and return messages if they are not yet implemented in subclass.
    """
    def __init__(self, data):
        self.data = data

    def validate_prior(self, prior):
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

    def __init__(self, data, prior):
        super(Gaussian, self).__init__(data)
        self.prior = prior
        self.validate_prior(prior)

def test(x, y):
    print('In test fn')
    return x + y