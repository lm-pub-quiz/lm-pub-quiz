import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Mapping, Type, Union, cast

import pandas as pd

log = logging.getLogger(__name__)


_registered_metric_classes: Dict[str, Type["RelationMetric"]] = {}


class MetricMetaClass(ABCMeta):
    def __new__(cls, *args, **kwargs):
        new_class = super().__new__(cls, *args, **kwargs)

        new_class = cast(Type["RelationMetric"], new_class)

        if hasattr(new_class, "metric_name"):
            _registered_metric_classes[new_class.metric_name] = new_class

        return new_class


MetricSpecification = Union[str, "RelationMetric", Type["RelationMetric"]]


class RelationMetric(metaclass=MetricMetaClass):
    metric_name: str

    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def add_instance(self, row: Mapping):
        pass

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        pass

    def add_instances_from_table(self, instance_table: pd.DataFrame):
        for _, row in instance_table.iterrows():
            self.add_instance(row)

    @classmethod
    def create_metric(cls, metric: MetricSpecification) -> "RelationMetric":
        if isinstance(metric, str):
            return _registered_metric_classes[metric]()

        elif isinstance(metric, RelationMetric):
            return metric

        elif issubclass(metric, RelationMetric):
            return metric()

        else:
            msg = f"Unknown type of metric: {type(metric)}"
            raise TypeError(msg)
