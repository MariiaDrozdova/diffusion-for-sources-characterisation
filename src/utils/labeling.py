import numpy as np

import torch


class_groups = {
    # group : indices (assuming 0th position is id)
    0: (),
    1: (1, 2, 3),
    2: (4, 5),
    3: (6, 7),
    4: (8, 9),
    5: (10, 11, 12, 13),
    6: (14, 15),
    7: (16, 17, 18),
    8: (19, 20, 21, 22, 23, 24, 25),
    9: (26, 27, 28),
    10: (29, 30, 31),
    11: (32, 33, 34, 35, 36, 37),
}


class_groups_indices = {g: np.array(ixs)-1 for g, ixs in class_groups.items()}


hierarchy = {
    # group : parent (group, label)
    2: (1, 1),
    3: (2, 1),
    4: (2, 1),
    5: (2, 1),
    7: (1, 0),
    8: (6, 0),
    9: (2, 0),
    10: (4, 0),
    11: (4, 0),
}


def make_galaxy_labels_hierarchical(labels: torch.Tensor) -> torch.Tensor:
    """ transform groups of galaxy label probabilities to follow the hierarchical order defined in galaxy zoo
    more info here: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview/the-galaxy-zoo-decision-tree
    labels is a NxL torch tensor, where N is the batch size and L is the number of labels,
    all labels should be > 1
    the indices of label groups are listed in class_groups_indices

    Return
    ------
    hierarchical_labels : NxL torch tensor, where L is the total number of labels
    """
    shift = labels.shape[1] > 37  # in case the id is included at 0th position, shift indices accordingly
    index = lambda i: class_groups_indices[i] + shift

    for i in range(1, 12):
        # normalize probabilities to 1
        norm = torch.sum(labels[:, index(i)], dim=1, keepdims=True)
        norm[norm == 0] += 1e-4   # add small number to prevent NaNs dividing by zero, yet keep track of gradient
        labels[:, index(i)] /= norm
        # renormalize according to hierarchical structure
        if i not in [1, 6]:
            parent_group_label = labels[:, index(hierarchy[i][0])]
            labels[:, index(i)] *= parent_group_label[:, hierarchy[i][1]].unsqueeze(-1)
    return labels
