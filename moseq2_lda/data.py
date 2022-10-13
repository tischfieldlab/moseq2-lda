from collections import Counter
from typing import Dict, List, Union
from dataclasses import dataclass

import numpy as np
from moseq2_viz.model.util import (get_syllable_statistics,
                                   get_transition_matrix, parse_model_results)
from moseq2_viz.util import parse_index
from sklearn import model_selection
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
    usages: np.ndarray
    frames: np.ndarray
    trans: np.ndarray

    def data(self, representation: Union['usages', 'frames', 'trans']) -> np.ndarray:
        return getattr(self, representation)

    @property
    def n_samples(self):
        return len(self.meta)

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

    def describe(self):
        ''' Describe the data within this instance
        '''
        print(f'{self.usages.shape[1]} modules in usages')
        print(f'{self.frames.shape[1]} modules in frames')
        print(f'{self.trans.shape[1]} transitions in trans')
        print()

        gcounts = Counter(self.groups)
        print(f'Breakdown of {self.n_samples} samples across {len(self.classes)} classes:')
        for cls in self.classes:
            print(f'{gcounts[cls]} {cls}')

        ''' Split this dataset into test and train subsets, in a stratified manner

        Parameters:
        k_fold (int): number of folds to split
        shuffle (bool): Whether to shuffle each classes samples before splitting
        seed (int|RandomState|None): when shuffle is True, affects the ordering of samples
        '''
        # split = model_selection.StratifiedKFold(n_splits=k_fold, shuffle=shuffle, random_state=seed)
        train_idx, test_idx, train_groups, test_groups = model_selection.train_test_split(np.arange(self.n_samples),
                                                                     self.groups,
                                                                     test_size=test_size,
                                                                     stratify=self.groups)

        # train_idx, test_idx = next(split.split(X=np.arange(self.n_samples), y=self.groups))
        print(train_idx, test_idx)

        for idx, g in zip(train_idx, train_groups):
            if self.meta[idx].group != g:
                raise(f'{self.meta[idx].group}, {g}, {idx}')

        train = MoseqRepresentations(
            meta=[self.meta[i] for i in train_idx],
            usages=self.usages[train_idx, :],
            frames=self.frames[train_idx, :],
            trans=self.trans[train_idx, :]
        )

        test = MoseqRepresentations(
            meta=[self.meta[i] for i in test_idx],
            usages=self.usages[test_idx, :],
            frames=self.frames[test_idx, :],
            trans=self.trans[test_idx, :]
        )
        print(train.groups, test.groups)

        return train, test


def load_representations(index_file: str, model_file: str, max_syllable: int = 100, groups: List[str] = None,
                         exclude_uuids: List[str] = None, prune_trans: bool = True) -> MoseqRepresentations:
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

    group_map = load_groups(index_file, groups)
    # print(group_map)

    if exclude_uuids is None:
        exclude_uuids = []

    tm_vals = []
    usage_vals = []
    frames_vals = []
    metadata_vals = []

    meta_keys_to_ignore = (
        'ColorDataType',
        'ColorResolution',
        'DepthDataType',
        'DepthResolution',
        'IsLittleEndian',
        'NidaqChannels',
        'NidaqSamplingRate'
    )

    for labs, grp, u in tqdm(list(zip(labels, label_group, model['keys'])), leave=False, disable=True):
        if (grp in group_map.keys()) and (u not in exclude_uuids):

            meta = sorted_index['files'][u]['metadata'].copy()
            for key_to_remove in meta_keys_to_ignore:
                meta.pop(key_to_remove, None)
            meta['uuid'] = u
            meta['group'] = group_map[grp]
            # print(f"'{grp}' -> '{group_map[grp]}'")
            metadata_vals.append(MoseqSampleMetadata(**meta))

            tm = get_transition_matrix([labs], combine=True, max_syllable=max_syllable)
            tm_vals.append(tm.ravel())

            u, _ = get_syllable_statistics(labs, count='usage')
            u_vals = np.array(list(u.values()))[:max_syllable+1]
            usage_vals.append(u_vals / np.sum(u_vals))

            f, _ = get_syllable_statistics(labs, count='frames')
            f_vals = np.array(list(f.values()))[:max_syllable+1]
            frames_vals.append(f_vals / np.sum(f_vals))

    tm_vals = np.array(tm_vals)
    if prune_trans:
        never_used_transitions = np.all(tm_vals == 0, axis=0)
        print(f'pruned {np.count_nonzero(never_used_transitions)} transitions which are never used')
        tm_vals = tm_vals[:, ~never_used_transitions]

    return MoseqRepresentations(
        meta=metadata_vals,
        usages=np.array(usage_vals),
        frames=np.array(frames_vals),
        trans=tm_vals
    )


def load_groups(index_file: str, custom_groupings: List[str] = None) -> Dict[str, str]:
    ''' Load available groups from a moseq2 index file, and return a map from origional group to (possibly custom) group

    Parameters:
    index_file (str): path to the moseq index file
    custom_groupings (List[str]): list of custom groupings, multiple subgroups should be comma separated

    Returns:
    Dict[str, str] - mapping of origional groups to possibly custom groups
    '''
    # Get group names available in model
    _, sorted_index = parse_index(index_file)
    available_groups = list(set([sorted_index['files'][uuid]['group'] for uuid in sorted_index['files'].keys()]))

    # { subgroup: supergroup }
    group_mapping: Dict[str, str] = {}

    if custom_groupings is None or len(custom_groupings) <= 0:
        for g in available_groups:
            group_mapping[g] = g

    else:
        for supergroup in custom_groupings:
            subgroups = supergroup.split(',')
            for subg in subgroups:
                if subg not in available_groups:
                    print(f'WARNING: subgroup "{subg}" from supergroup "{supergroup}" not found in model! Omitting...')
                    continue

                if subg in group_mapping:
                    print(f'WARNING: subgroup "{subg}" from supergroup "{supergroup}" already registered '
                          f'to supergroup "{group_mapping[subg]}"! Omitting...')
                    continue

                group_mapping[subg] = supergroup

    return group_mapping
