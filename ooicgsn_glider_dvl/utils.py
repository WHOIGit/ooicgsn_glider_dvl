#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Andrew Reed
@brief Provides shared methods for working with glider datasets
"""

import numpy as np

def unix_time(dt, unit='s'):
    """Convert numpy datetime64 object to unix time"""
    return (dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, unit)