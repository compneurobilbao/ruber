# -*- coding: utf-8 -*-
"""
Configuration manager for workflow definitions.
This works over Kaptan:https://github.com/emre/kaptan

The global configuration registry is declared in the bottom of this file.
"""
import os.path as op

from   nipype.pipeline.engine import Node, MapNode, JoinNode
from   nipype.interfaces.base import isdefined



def _check_file(file_path):
    fpath = op.abspath(op.expanduser(file_path))

    if not op.isfile(fpath):
        raise IOError("Could not find configuration file {}.".format(fpath))


def node_settings(node_name):
    global PYPES_CFG
    for k, v in PYPES_CFG.items():
        if k.startswith(node_name):
            yield k, v


def update_config(value):
    """ Value can be a configuration file path or a dictionary with
    configuration settings."""
    global PYPES_CFG
    if isinstance(value, str):
        PYPES_CFG.update_from_file(value)
    elif isinstance(value, dict):
        PYPES_CFG.update(value)
    else:
        raise NotImplementedError('Cannot update the configuration with {}.'.format(value))


def _set_node_inputs(node, params, overwrite=False):
    for k, v in params.items():
        try:
            if not isdefined(getattr(node.inputs, k)):
                setattr(node.inputs, k, v)
            else:
                if overwrite:
                    setattr(node.inputs, k, v)
        except AttributeError as ate:
            raise AttributeError('Error in configuration settings: node `{}` '
                                 'has no attribute `{}`.'.format(node, k)) from ate


def _get_params_for(node_name):
    pars = {}
    for k, v in node_settings(node_name):
        nuk = '.'.join(k.split('.')[1:]) if '.' in k else k
        pars[nuk] = v

    return pars


def check_mandatory_inputs(node_names):
    """ Raise an exception if any of the items in the List[str] `node_names` is not
    present in the global configuration settings."""
    for name in node_names:
        if name not in PYPES_CFG:
            raise AttributeError('Could not find a configuration parameter for {}. '
                                 'Please set it in the an input configuration file.'.format(name))


def get_config_setting(param_name, default=''):
    """ Return the value for the entry with name `param_name` in the global configuration."""
    return PYPES_CFG.get(param_name, default=default)


def setup_node(interface, name, settings=None, overwrite=True, **kwargs):
    """ Create a pe.Node from `interface` with a given name.
    Check in the global configuration if there is any value for the node name and will set it.

    Parameters
    ----------
    interface: nipype.interface

    name: str

    settings: dict
        Dictionary with values for the pe.Node inputs.
        These will have higher priority than the ones in the global Configuration.

    overwrite: bool
        If True will overwrite the settings of the node if they are already defined.
        Default: True

    kwargs: keyword arguments
        type: str or None.
            choices: 'map', 'join, or None.
            If 'map' will return a MapNode.
            If 'join' will return a JoinNode.
            If None will return a Node.

        Extra arguments to pass to nipype.Node __init__ function.


    Returns
    -------
    node: nipype.Node
    """
    typ = kwargs.pop('type', None)
    if typ == 'map':
        node_class = MapNode
    elif typ == 'join':
        node_class = JoinNode
    else:
        node_class = Node
    node = node_class(interface=interface, name=name, **kwargs)

    params = _get_params_for(name)
    if settings is not None:
        params.update(settings)

    _set_node_inputs(node, params, overwrite=overwrite)

    return node


# ---------------------------------------------------------------------------
# Helper functions for specific parameters of config
# ---------------------------------------------------------------------------
def check_atlas_file():
    """ Return True and the path to the atlas_file if the configuration settings
    `normalize_atlas` is True and `atlas_file` points to an existing file.
    If `normalize_atlas` is False will return False and an empty string.
    Otherwise will raise a FileNotFoundError.

    Returns
    -------
    do_atlas: bool

    atlas_file: str
        Existing file path to the atlas file

    Raises
    ------
    FileNotFoundError
        If the `normalize_atlas` option is True but the
        `atlas_file` is not an existing file.
    """
    normalize_atlas = get_config_setting('normalize_atlas', default=False)
    if not normalize_atlas:
        return False, ''

    atlas_file = get_config_setting('atlas_file', default='')
    if not op.isfile(atlas_file):
        raise FileNotFoundError('Could not find atlas file in {}. '
                                'Please set `normalize_atlas` to False '
                                'or give an existing atlas image.'.format(atlas_file))
    return True, atlas_file

