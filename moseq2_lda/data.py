from dataclasses import dataclass
from typing import List

import numpy as np
from moseq2_viz.model.util import (get_syllable_statistics,
                                   get_transition_matrix, parse_model_results)
from moseq2_viz.util import parse_index
from tqdm import tqdm


@dataclass
class MoseqSampleMetadata:
    ''' Metadata for a single sample
    '''
    uuid: str
    group: str
    SessionName: str
    SubjectName: str
    StartTime: str
    ApparatusName: str

@dataclass
class MoseqRepresentations:
    ''' Contains various representations of moseq data
    '''
    meta: List[MoseqSampleMetadata]
    usage: np.ndarray
    frames: np.ndarray
    trans: np.ndarray

    @property
    def classes(self):
        ''' Gets the unique set of group labels in the dataset '''
        return list(set(self.groups))

    @property
    def groups(self):
        ''' Gets an array of group labels for each sample in the dataset '''
        return np.array([m.group for m in self.meta])

    @property
    def uuids(self):
        ''' Gets an array of uuids for each sample in the dataset '''
        return np.array([m.uuid for m in self.meta])


def load_representations(index_file: str, model_file: str, max_syllable: int=100, groups: List[str]=None, exclude_uuids: List[str]=None, prune_trans: bool=True) -> MoseqRepresentations:
    ''' Load representations of moseq data

    Parameters:
    index_file (str): path to the moseq index file
    model_file (str): path to the moseq model file
    max_syllable (int): maximum syllable id to consider
    groups (List[str]|None): if None, consider all groups in the model, otherwise restrict output to only these groups
    exclude_uuids (List[str]|None): if None, consider all samples in the model, otherwise, exclude samples with these uuids
    prune_trans (bool): if True, prune transitions in which all groups have a value of zero, otherwise consider all transitions

    Returns:
    MoseqRepresentations - representations of moseq data
    '''
    _, sorted_index = parse_index(index_file)
    model = parse_model_results(model_file, sort_labels_by_usage=True, count='usage')

    labels = model['labels']
    label_group = [sorted_index['files'][uuid]['group'] for uuid in model['keys']]

    if groups is None:
        groups = list(set(label_group))

    if exclude_uuids is None:
        exclude_uuids = []

    tm_vals = []
    usage_vals = []
    frames_vals = []
    metadata_vals = []

    for l, g, u in tqdm(list(zip(labels, label_group, model['keys'])), leave=False):
        if (g in groups) and (u not in exclude_uuids):

            meta = sorted_index['files'][u]['metadata'].copy()
            for key_to_remove in ('ColorDataType', 'ColorResolution', 'DepthDataType', 'DepthResolution', 'IsLittleEndian', 'NidaqChannels', 'NidaqSamplingRate'):
                meta.pop(key_to_remove, None)
            meta['uuid'] = u
            meta['group'] = g
            metadata_vals.append(MoseqSampleMetadata(**meta))

            tm = get_transition_matrix([l], combine=True, max_syllable=max_syllable)
            tm_vals.append(tm.ravel())

            u, _ = get_syllable_statistics(l, count='usage')
            u_vals = np.array(list(u.values()))[:max_syllable+1]
            usage_vals.append(u_vals / np.sum(u_vals))

            f, _ = get_syllable_statistics(l, count='frames')
            f_vals = np.array(list(f.values()))[:max_syllable+1]
            frames_vals.append(f_vals / np.sum(f_vals))

    tm_vals = np.array(tm_vals)
    if prune_trans:
        never_used_transitions = np.all(tm_vals == 0, axis=0)
        print(f'pruned {np.count_nonzero(never_used_transitions)} transitions which are never used')
        tm_vals = tm_vals[:, ~never_used_transitions]

    return MoseqRepresentations(
        meta=metadata_vals,
        usage=np.array(usage_vals),
        frames=np.array(frames_vals),
        trans=tm_vals
    )
