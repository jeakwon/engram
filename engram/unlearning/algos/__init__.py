# from .boundary_ex import boundary_expanding
# from .boundary_sh import boundary_shrink
from .fisher import fisher, fisher_new
from .FT import FT, FT_l1
from .GA import GA, GA_l1

from .RL import RL

# from .Wfisher import Wfisher
from .neggrad import negative_grad

from .RL_original import RL_og

# from .scrub import scrub


def raw(data_loaders, model, criterion, args, mask=None):
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "GA":
        return GA
    elif name == "GA_l1":
        return GA_l1
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "NG":
        return negative_grad
    elif name == "fisher":
        return fisher
    elif name == "fisher_new":
        return fisher_new
    elif name == "RL":
        return RL
    elif name == "RL_og":
        return RL_og
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")

    """
    if name == "raw":
        return raw
    elif name == "RL":
        return RL
    elif name == "RL_og":
        return RL_og
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "fisher":
        return fisher
    elif name == "retrain":
        return retrain
    elif name == "fisher_new":
        return fisher_new
    elif name == "wfisher":
        return Wfisher
    elif name == "GA_l1":
        return GA_l1
    elif name == "boundary_expanding":
        return boundary_expanding
    elif name == "boundary_shrink":
        return boundary_shrink
    elif name == "NG":
        return negative_grad
    elif name == "SCRUB":
        return scrub
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")

    """
