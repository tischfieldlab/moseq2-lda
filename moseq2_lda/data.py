from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Union
from typing_extensions import Literal

import numpy as np
from moseq2_viz.model.util import (get_syllable_statistics,
                                   get_transition_matrix, parse_model_results)
from moseq2_viz.util import parse_index
from sklearn import model_selection
from tqdm import tqdm


@dataclass(frozen=True)
class MoseqSampleMetadata:
    ''' Dataclass storing metadata for a single sample
    '''

    ''' Unique identifier for this sample '''
    uuid: str
    ''' Group to which this sample belongs '''
    group: str
    ''' Session Name given to this sample '''
    SessionName: str
    ''' Subject Name given to this sample '''
    SubjectName: str
    ''' Date and time at which this sample's raw data was collected '''
    StartTime: str
    ''' Name of the apparatus in which this sample's raw data was collected'''
    ApparatusName: str


RepresentationType = Literal['usages', 'frames', 'trans']


@dataclass(frozen=True)
class MoseqRepresentations:
    ''' Contains various representations of moseq data
    '''
    ''' Collection of metadata associated with the moseq data'''
    meta: List[MoseqSampleMetadata]
    ''' Usages representation (relative emission rates), of shape (nsamples, nmodules) '''
    usages: np.ndarray
    ''' frames representation (relative frame counts), of shape (nsamples, nmodules) '''
    frames: np.ndarray
    ''' Transitions (bigram normalized transition probabilities), of shape (nsamples, ntransitions) '''
    trans: np.ndarray

    def data(self, representation: RepresentationType) -> np.ndarray:
        ''' Gets the data for a given representation

        Parameters:
        representation (RepresentationType): Type of representation to yield

        Returns:
        np.ndarray - numpy array containing data of the requested representation type
        '''
        return getattr(self, representation)

    @property
    def n_samples(self):
        ''' Gets the number of samples in this dataset'''
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

    def describe(self, as_str: bool = False) -> Union[str, None]:
        ''' Describe the data within this instance

        Parameters:
        as_str (bool): if True, return the description as a string, otherwise print to stdout

        Returns:
        None if `as_str` is False, otherwise the description as a string
        '''
        buffer = ''
        buffer += f'{self.usages.shape[1]} modules in usages\n'
        buffer += f'{self.frames.shape[1]} modules in frames\n'
        buffer += f'{self.trans.shape[1]} transitions in trans\n'
        buffer += '\n'

        gcounts = Counter(self.groups)
        buffer += f'Breakdown of {self.n_samples} samples across {len(self.classes)} classes:\n'
        for cls in self.classes:
            buffer += f'{gcounts[cls]} {cls}\n'

        buffer += '\n'

        if as_str:
            return buffer
        else:
            print(buffer)
            return None

    def split(self, test_size: float = 0.3, seed: Union[int, np.random.RandomState, None] = None):
        ''' Split this dataset into test and train subsets, in a stratified manner

        Parameters:
        test_size (int): percentage of data to be used in the test subset, `1 - test_size` will be used for the train subset
        seed (int|RandomState|None): when shuffle is True, affects the ordering of samples

        Returns:
        Tuple[MoseqRepresentations, MoseqRepresentations] - train and test subsets, respectively
        '''
        train_idx, test_idx, train_groups, test_groups = model_selection.train_test_split(np.arange(self.n_samples),
                                                                                          self.groups,
                                                                                          test_size=test_size,
                                                                                          stratify=self.groups,
                                                                                          shuffle=True,
                                                                                          random_state=seed)

        # sanity check: make sure groups match in train subset
        for idx, g in zip(train_idx, train_groups):
            if self.meta[idx].group != g:
                raise ValueError(f'Failed train grouping sanity check: {self.meta[idx].group}, {g}, {idx}')

        # sanity check: make sure groups match in test subset
        for idx, g in zip(test_idx, test_groups):
            if self.meta[idx].group != g:
                raise ValueError(f'Failed train grouping sanity check: {self.meta[idx].group}, {g}, {idx}')

        # build the train subset
        train = MoseqRepresentations(
            meta=[self.meta[i] for i in train_idx],
            usages=self.usages[train_idx, :],
            frames=self.frames[train_idx, :],
            trans=self.trans[train_idx, :]
        )

        # build the test subset
        test = MoseqRepresentations(
            meta=[self.meta[i] for i in test_idx],
            usages=self.usages[test_idx, :],
            frames=self.frames[test_idx, :],
            trans=self.trans[test_idx, :]
        )

        # return train and test subsets of this data
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

    tm_vals_np = np.array(tm_vals)
    if prune_trans:
        never_used_transitions = np.all(tm_vals_np == 0, axis=0)
        print(f'pruned {np.count_nonzero(never_used_transitions)} transitions which are never used')
        tm_vals_np = tm_vals_np[:, ~never_used_transitions]

    return MoseqRepresentations(
        meta=metadata_vals,
        usages=np.array(usage_vals),
        frames=np.array(frames_vals),
        trans=tm_vals_np
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
