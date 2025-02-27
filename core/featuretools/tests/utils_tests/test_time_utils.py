from itertools import chain

import numpy as np
import pandas as pd
import pytest

from core.featuretools.utils import convert_time_units, make_temporal_cutoffs


def test_make_temporal_cutoffs():
    instance_ids = pd.Series(range(10))
    cutoffs = pd.date_range(start='1/2/2015', periods=10, freq='1d')
    temporal_cutoffs_by_nwindows = make_temporal_cutoffs(instance_ids,
                                                         cutoffs,
                                                         window_size='1h',
                                                         num_windows=2)

    assert temporal_cutoffs_by_nwindows.shape[0] == 20
    actual_instances = chain.from_iterable([[i, i] for i in range(10)])
    actual_times = [
        '1/1/2015 23:00:00',
        '1/2/2015 00:00:00',
        '1/2/2015 23:00:00',
        '1/3/2015 00:00:00',
        '1/3/2015 23:00:00',
        '1/4/2015 00:00:00',
        '1/4/2015 23:00:00',
        '1/5/2015 00:00:00',
        '1/5/2015 23:00:00',
        '1/6/2015 00:00:00',
        '1/6/2015 23:00:00',
        '1/7/2015 00:00:00',
        '1/7/2015 23:00:00',
        '1/8/2015 00:00:00',
        '1/8/2015 23:00:00',
        '1/9/2015 00:00:00',
        '1/9/2015 23:00:00',
        '1/10/2015 00:00:00',
        '1/10/2015 23:00:00',
        '1/11/2015 00:00:00',
        '1/11/2015 23:00:00'
    ]
    actual_times = [pd.Timestamp(c) for c in actual_times]

    for computed, actual in zip(temporal_cutoffs_by_nwindows['instance_id'], actual_instances):
        assert computed == actual
    for computed, actual in zip(temporal_cutoffs_by_nwindows['time'], actual_times):
        assert computed == actual

    cutoffs = [pd.Timestamp('1/2/2015')] * 9 + [pd.Timestamp('1/3/2015')]
    starts = [pd.Timestamp('1/1/2015')] * 9 + [pd.Timestamp('1/2/2015')]
    actual_times = ['1/1/2015 00:00:00', '1/2/2015 00:00:00'] * 9
    actual_times += ['1/2/2015 00:00:00', '1/3/2015 00:00:00']
    actual_times = [pd.Timestamp(c) for c in actual_times]
    temporal_cutoffs_by_wsz_start = make_temporal_cutoffs(instance_ids,
                                                          cutoffs,
                                                          window_size='1d',
                                                          start=starts)

    for computed, actual in zip(temporal_cutoffs_by_wsz_start['instance_id'], actual_instances):
        assert computed == actual
    for computed, actual in zip(temporal_cutoffs_by_wsz_start['time'], actual_times):
        assert computed == actual

    cutoffs = [pd.Timestamp('1/2/2015')] * 9 + [pd.Timestamp('1/3/2015')]
    starts = [pd.Timestamp('1/1/2015')] * 10
    actual_times = ['1/1/2015 00:00:00', '1/2/2015 00:00:00'] * 9
    actual_times += ['1/1/2015 00:00:00', '1/3/2015 00:00:00']
    actual_times = [pd.Timestamp(c) for c in actual_times]
    temporal_cutoffs_by_nw_start = make_temporal_cutoffs(instance_ids,
                                                         cutoffs,
                                                         num_windows=2,
                                                         start=starts)

    for computed, actual in zip(temporal_cutoffs_by_nw_start['instance_id'], actual_instances):
        assert computed == actual
    for computed, actual in zip(temporal_cutoffs_by_nw_start['time'], actual_times):
        assert computed == actual


def test_convert_time_units():
    units = {'years': 31540000, 'months': 2628000, 'days': 86400, 'hours': 3600, 'minutes': 60, 'seconds': 1, 'milliseconds': 0.001, 'nanoseconds': 0.000000001}
    for each in units:
        assert (convert_time_units(units[each] * 2, each) == 2)
        assert np.isclose(convert_time_units(float(units[each] * 2), each), 2)

    error_text = "Invalid unit given, make sure it is plural"
    with pytest.raises(ValueError, match=error_text):
        convert_time_units("jnkwjgn", 10)
