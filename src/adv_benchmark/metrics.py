"""
Module that contains useful metrics used througout the project
"""

import numpy as np


def degree_of_change(x_adv_list, x_list):
    """
    This fonction computes the relative difference in % between the input image and
    the adv image. The lower this DOC is, the more subtle the adv image is
    inputs:
    -x_adv_list (list on images array): list of adv images
    -x_list (list on images array): list of adv images
    outputs:
    -Degree of change (float)
    """
    list_len = len(x_adv_list)
    total = 0
    for i in range(list_len):
        total += (
            100
            * np.linalg.norm(np.reshape(x_adv_list[i] - x_list[i], -1), ord=1)
            / np.linalg.norm(np.reshape(x_list[i], -1), ord=1)
        )
    return total / list_len


def success_rate(success_list):
    """
    this fonction returns the ratio of adversarial examples that have fooled a model
    inputs:
    -success_list (list of booleans): true if this example have fooled the model, False otherwise
    output:
    -success rate (float between 0 and 1)

    """
    success = len([i for i in success_list if i])
    total = len(success_list)
    return success / total
