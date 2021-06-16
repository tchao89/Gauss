from core.featuretools.primitives.base import make_agg_primitive
from core.featuretools.variable_types import Numeric

CustomSum = make_agg_primitive(lambda x: sum(x),
                               name="CustomSum",
                               input_types=[Numeric],
                               return_type=Numeric)
