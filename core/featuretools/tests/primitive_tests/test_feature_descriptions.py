import json
import os

import pytest

from core.featuretools import describe_feature
from core.featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature
)
from core.featuretools.primitives import (
    Absolute,
    AggregationPrimitive,
    CumMean,
    EqualScalar,
    Mean,
    Mode,
    NMostCommon,
    NumUnique,
    PercentTrue,
    Sum,
    TransformPrimitive
)
from core.featuretools.variable_types import Discrete


def test_identity_description(es):
    feature = IdentityFeature(es['log']['session_id'])
    description = 'The "session_id".'

    assert describe_feature(feature) == description


def test_direct_description(es):
    feature = DirectFeature(es['customers']['loves_ice_cream'], es['sessions'])
    description = 'The "loves_ice_cream" for the instance of "customers" associated ' \
                  'with this instance of "sessions".'
    assert describe_feature(feature) == description

    deep_direct = DirectFeature(feature, es['log'])
    deep_description = 'The "loves_ice_cream" for the instance of "customers" ' \
                       'associated with the instance of "sessions" associated with ' \
                       'this instance of "log".'
    assert describe_feature(deep_direct) == deep_description

    agg = AggregationFeature(es['log']['purchased'], es['sessions'], PercentTrue)
    complicated_direct = DirectFeature(agg, es['log'])
    agg_on_direct = AggregationFeature(complicated_direct, es['products'], Mean)

    complicated_description = 'The average of the percentage of true values in ' \
        'the "purchased" of all instances of "log" for each "id" in "sessions" for ' \
        'the instance of "sessions" associated with this instance of "log" of all ' \
        'instances of "log" for each "id" in "products".'
    assert describe_feature(agg_on_direct) == complicated_description


def test_transform_description(es):
    feature = TransformFeature(es['log']['value'], Absolute)
    description = 'The absolute value of the "value".'
    assert describe_feature(feature) == description


def test_groupby_transform_description(es):
    feature = GroupByTransformFeature(es['log']['value'], CumMean, es['log']['session_id'])
    description = 'The cumulative mean of the "value" for each "session_id".'

    assert describe_feature(feature) == description


def test_aggregation_description(es):
    feature = AggregationFeature(es['log']['value'], es['sessions'], Mean)
    description = 'The average of the "value" of all instances of "log" for each "id" in "sessions".'
    assert describe_feature(feature) == description

    stacked_agg = AggregationFeature(feature, es['customers'], Sum)
    stacked_description = 'The sum of t{} of all instances of "sessions" for each "id" ' \
                          'in "customers".'.format(description[1:-1])
    assert describe_feature(stacked_agg) == stacked_description


def test_aggregation_description_where(es):
    where_feature = TransformFeature(es['log']['countrycode'], EqualScalar('US'))
    feature = AggregationFeature(es['log']['value'], es['sessions'],
                                 Mean, where=where_feature)
    description = 'The average of the "value" of all instances of "log" where the ' \
                  '"countrycode" is US for each "id" in "sessions".'

    assert describe_feature(feature) == description


def test_aggregation_description_use_previous(es):
    feature = AggregationFeature(es['log']['value'], es['sessions'],
                                 Mean, use_previous='5d')
    description = 'The average of the "value" of the previous 5 days of "log" for each "id" in "sessions".'

    assert describe_feature(feature) == description


def test_multioutput_description(es):
    n_most_common = NMostCommon(2)
    n_most_common_feature = AggregationFeature(es['log']['zipcode'], es['sessions'], n_most_common)
    first_most_common_slice = n_most_common_feature[0]
    second_most_common_slice = n_most_common_feature[1]

    n_most_common_base = 'The 2 most common values of the "zipcode" of all instances of "log" for each "id" in "sessions".'
    n_most_common_first = 'The most common value of the "zipcode" of all instances of "log" ' \
                          'for each "id" in "sessions".'
    n_most_common_second = 'The 2nd most common value of the "zipcode" of all instances of ' \
                           '"log" for each "id" in "sessions".'

    assert describe_feature(n_most_common_feature) == n_most_common_base
    assert describe_feature(first_most_common_slice) == n_most_common_first
    assert describe_feature(second_most_common_slice) == n_most_common_second

    class CustomMultiOutput(TransformPrimitive):
        name = "custom_multioutput"
        input_types = [Discrete]
        return_type = Discrete

        number_output_features = 4

    custom_feat = TransformFeature(es['log']['zipcode'], CustomMultiOutput)

    generic_base = 'The result of applying CUSTOM_MULTIOUTPUT to the "zipcode".'
    generic_first = 'The 1st output from applying CUSTOM_MULTIOUTPUT to the "zipcode".'
    generic_second = 'The 2nd output from applying CUSTOM_MULTIOUTPUT to the "zipcode".'

    assert describe_feature(custom_feat) == generic_base
    assert describe_feature(custom_feat[0]) == generic_first
    assert describe_feature(custom_feat[1]) == generic_second

    CustomMultiOutput.description_template = ['the multioutput of {}',
                                              'the {nth_slice} multioutput part of {}']
    template_base = 'The multioutput of the "zipcode".'
    template_first_slice = 'The 1st multioutput part of the "zipcode".'
    template_second_slice = 'The 2nd multioutput part of the "zipcode".'
    template_third_slice = 'The 3rd multioutput part of the "zipcode".'
    template_fourth_slice = 'The 4th multioutput part of the "zipcode".'
    assert describe_feature(custom_feat) == template_base
    assert describe_feature(custom_feat[0]) == template_first_slice
    assert describe_feature(custom_feat[1]) == template_second_slice
    assert describe_feature(custom_feat[2]) == template_third_slice
    assert describe_feature(custom_feat[3]) == template_fourth_slice

    CustomMultiOutput.description_template = ['the multioutput of {}',
                                              'the primary multioutput part of {}',
                                              'the secondary multioutput part of {}']
    custom_base = 'The multioutput of the "zipcode".'
    custom_first_slice = 'The primary multioutput part of the "zipcode".'
    custom_second_slice = 'The secondary multioutput part of the "zipcode".'
    bad_slice_error = 'Slice out of range of template'
    assert describe_feature(custom_feat) == custom_base
    assert describe_feature(custom_feat[0]) == custom_first_slice
    assert describe_feature(custom_feat[1]) == custom_second_slice
    with pytest.raises(IndexError, match=bad_slice_error):
        describe_feature(custom_feat[2])


def test_generic_description(es):
    class NoName(TransformPrimitive):
        input_types = [Discrete]
        output_type = Discrete

        def generate_name(self, base_feature_names):
            return u"%s(%s%s)" % (
                'NO_NAME',
                u", ".join(base_feature_names),
                self.get_args_string(),
            )

    class CustomAgg(AggregationPrimitive):
        name = 'custom_aggregation'
        input_types = [Discrete]
        output_type = Discrete

    class CustomTrans(TransformPrimitive):
        name = 'custom_transform'
        input_types = [Discrete]
        output_type = Discrete

    no_name = TransformFeature(es['log']['zipcode'], NoName)
    no_name_description = 'The result of applying NoName to the "zipcode".'
    assert describe_feature(no_name) == no_name_description

    custom_agg = AggregationFeature(es['log']['zipcode'], es['customers'], CustomAgg)
    custom_agg_description = 'The result of applying CUSTOM_AGGREGATION to the "zipcode" of all instances of "log" for each "id" in "customers".'
    assert describe_feature(custom_agg) == custom_agg_description

    custom_trans = TransformFeature(es['log']['zipcode'], CustomTrans)
    custom_trans_description = 'The result of applying CUSTOM_TRANSFORM to the "zipcode".'
    assert describe_feature(custom_trans) == custom_trans_description


def test_variable_description(es):
    variable_description = 'the name of the device used for each session'
    es['sessions']['device_name'].description = variable_description
    identity_feat = IdentityFeature(es['sessions']['device_name'])
    assert describe_feature(identity_feat) == variable_description[0].upper() + variable_description[1:] + '.'


def test_metadata(es, tmpdir):
    identity_feature_descriptions = {'sessions: device_name': 'the name of the device used for each session',
                                     'customers: id': "the customer's id"}
    agg_feat = AggregationFeature(es['sessions']['device_name'], es['customers'], NumUnique)
    agg_description = 'The number of unique elements in the name of the device used for each '\
                      'session of all instances of "sessions" for each customer\'s id.'
    assert describe_feature(agg_feat, feature_descriptions=identity_feature_descriptions) == agg_description

    transform_feat = GroupByTransformFeature(es['log']['value'], CumMean, es['log']['session_id'])
    transform_description = 'The running average of the "value" for each "session_id".'
    primitive_templates = {"cum_mean": "the running average of {}"}
    assert describe_feature(transform_feat, primitive_templates=primitive_templates) == transform_description

    custom_agg = AggregationFeature(es['log']['zipcode'], es['sessions'], Mode)
    auto_description = 'The most frequently occurring value of the "zipcode" of all instances of "log" for each "id" in "sessions".'
    custom_agg_description = "the most frequently used zipcode"
    custom_feature_description = custom_agg_description[0].upper() + custom_agg_description[1:] + '.'
    feature_description_dict = {'sessions: MODE(log.zipcode)': custom_agg_description}
    assert describe_feature(custom_agg) == auto_description
    assert describe_feature(custom_agg, feature_descriptions=feature_description_dict) == custom_feature_description

    metadata = {
        'feature_descriptions': {**identity_feature_descriptions, **feature_description_dict},
        'primitive_templates': primitive_templates
    }
    metadata_path = os.path.join(tmpdir, 'description_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    assert describe_feature(agg_feat, metadata_file=metadata_path) == agg_description
    assert describe_feature(transform_feat, metadata_file=metadata_path) == transform_description
    assert describe_feature(custom_agg, metadata_file=metadata_path) == custom_feature_description
