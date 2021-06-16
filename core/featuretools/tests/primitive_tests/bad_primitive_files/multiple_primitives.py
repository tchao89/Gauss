from core.featuretools.primitives import make_agg_primitive
from core.featuretools.variable_types import Numeric

CustomMax = make_agg_primitive(lambda x: max(x),
                               name="CustomMax",
                               input_types=[Numeric],
                               return_type=Numeric)

CustomSum = make_agg_primitive(lambda x: sum(x),
                               name="CustomSum",
                               input_types=[Numeric],
                               return_type=Numeric)
