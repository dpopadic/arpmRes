# -*- coding: utf-8 -*-
import numpy as np


def sector_select(sectors, sect):
    """For details, see here.

    Parameters
    ----------
         sectors : string array
         sect : string array

    Returns
    -------
        index : array

    """

    a = sect == sectors
    index = np.where(a)[0]
    return index
