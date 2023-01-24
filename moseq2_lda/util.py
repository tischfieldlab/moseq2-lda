"""Module contains utility functions."""
import collections

import click


def dict_merge(dct: dict, merge_dct: dict):
    """Recursive dict merge.

    Inspired by `dict.update()`, instead of updating only top-level keys,
    dict_merge recurses down into dicts nested to an arbitrary depth,
    updating keys. The `merge_dct` is merged into `dct`.

    Args:
        dct: dict onto which the merge is executed
        merge_dct: dict merged into dct
    """
    for k, v in merge_dct.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.Mapping):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def click_monkey_patch_option_show_defaults():
    """Monkey patch click.core.Option to turn on showing default values."""
    orig_init = click.core.Option.__init__

    def new_init(self, *args, **kwargs):
        """This version of click.core.Option.__init__ will set show default values to True."""
        orig_init(self, *args, **kwargs)
        self.show_default = True

    # end new_init()
    click.core.Option.__init__ = new_init  # type: ignore
