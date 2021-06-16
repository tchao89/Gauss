from core.featuretools.primitives.base import make_agg_primitive
from core.featuretools.variable_types import Numeric

CustomMax = make_agg_primitive(lambda x: max(x),
                               name="CustomMax",
                               input_types=[Numeric],
                               return_type=Numeric)
