import os
from models.coop import coop
from models.models import get_csp, get_hpl

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_model(train_dataset, config, device):
    if config.experiment_name == "coop":
        return coop(train_dataset, config, device)

    elif config.experiment_name == "csp":
        return get_csp(train_dataset, config, device)

    elif config.experiment_name == "hpl":
        return get_hpl(train_dataset, config, device)

    else:
        raise NotImplementedError(
            "Error: Unrecognized Experiment Name {:s}.".format(
                config.experiment_name
            )
        )
