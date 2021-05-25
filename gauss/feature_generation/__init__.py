from gauss.feature_generation.featuretools_generation import featuretools as ft

DATATYPEDICT = {
    "bool": ft.variable_types.Boolean,
    "numeric": ft.variable_types.Numeric,
    "categorical": ft.variable_types.Categorical,
    "ordinal": ft.variable_types.Ordinal
}
