"""Dict util functions"""


def _pack_dicts_rec(dicts, keys, values):
    """
    Add multiple values to multiple dicts using several orders of keys
    so that value[i] will be set as dicts[i][keys[0]][keys[1]][...][keys[M-1]]
    This is a generalized version of pack_dicts() that accepts a list of keys
    :param dicts: list of N dicts
    :param keys: list of M keys
    :param values: list of N values; has to have same length as dicts;
    """
    for dict_, value in zip(dicts, values):
        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                dict_[key] = value
            else:
                if key not in dict_:
                    dict_[key] = {}
                dict_ = dict_[key]


def pack_dicts(dicts, key, values):
    """
    Add multiple values to multiple dicts using the same key
    so that value[i] will be set as dicts[i][key]
    :param dicts: list of dicts
    :param key: key
    :param values: list of values; has to have same length as dicts;
    """
    if isinstance(key, (list, tuple)):
        return _pack_dicts_rec(dicts, key, values)
    for dict_, value in zip(dicts, values):
        dict_[key] = value


def unpack_dicts(dicts, key):
    """
    Retrieve multiple values from multiple dicts using the same key
    :param dicts: list of dicts
    :param key: key
    :return: list of values of same length as dicts, where values[i] = dicts[i][key]
    """
    return [dict_[key] for dict_ in dicts]


def init_dicts(num_dicts):
    """
    Create a list of empty dicts
    :param num_dicts: number of dicts
    :return: list of num_dicts dicts
    """
    return [{} for _ in range(num_dicts)]
