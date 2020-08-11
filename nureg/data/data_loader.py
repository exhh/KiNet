from __future__ import print_function

data_params = {}
def register_data_params(name):
    def decorator(cls):
        data_params[name] = cls
        return cls
    return decorator

dataset_obj = {}
def register_dataset_obj(name):
    def decorator(cls):
        dataset_obj[name] = cls
        return cls
    return decorator


class DatasetParams(object):
    "Class variables defined."
    num_channels = 1
    image_size   = 16
    mean         = 0.1307
    std          = 0.3081
    num_cls      = 10
    target_transform = None

def get_fcn_dataset(name, rootdir, **kwargs):
    return dataset_obj[name](rootdir, **kwargs)
