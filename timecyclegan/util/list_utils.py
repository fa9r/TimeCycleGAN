def append_to_lists(lists, values):
    for list_, value in zip(lists, values):
        list_.append(value)


def get_from_lists(lists, index=None, index_start=None, index_end=None):
    assert index is not None or index_start is not None or index_end is not None
    if index is not None:
        assert index_start is None and index_end is None
        index_start, index_end = index, index + 1
    if index_start is None:
        return [list_[:index_end] for list_ in lists]
    if index_end is None:
        return [list_[index_start:] for list_ in lists]
    return [list_[index_start:index_end] for list_ in lists]
