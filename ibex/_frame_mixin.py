from __future__ import absolute_import

import operator
import uuid

from sklearn import pipeline


class FrameMixin(object):
    """
    A base class for steps taking pandas entities, not
        numpy entities.

    Subclass this step to indicate that a step takes pandas
        entities.
    """

    def __init__(self, columns=None):
        if columns is not None:
            self.set_paams(columns=columns)

    def set_params(self, **params):
        if 'columns' not in params:
            return

        if not hasattr(self, '__cols'):
            self.__cols = params['columns']

        if set(params['columns']) != set(self.__cols):
            raise KeyError()

    def get_params(self, deep=True):
        params = {}
        try:
            params['columns'] = self.__cols
        except AttributeError:
            pass
        return params

    @classmethod
    def is_subclass(cls, step):
        """
        Returns:
            Whether a step is a subclass of Stage.

        Arguments:
            step: A Stage or a pipeline.
        """
        if issubclass(type(step), pipeline.Pipeline):
            if not step.steps:
                raise ValueError('Cannot use 0-length pipeline')
            return cls.is_subclass(step.steps[0][1])
        return issubclass(type(step), FrameMixin)

    def __or__(self, other):
        if issubclass(type(other), pipeline.Pipeline):
            others = [operator.itemgetter(1)(e) for e in other.steps]
        else:
            others = [other]

        return pipeline.make_pipeline(self, *others)

    def __ror__(self, other):
        if issubclass(type(other), pipeline.Pipeline):
            others = [operator.itemgetter(1)(e) for e in other.steps]
        else:
            others = [other]
        others += [self]

        return pipeline.make_pipeline(*others)

    def __add__(self, other):
        from ._feature_union import FeatureUnion

        if isinstance(self, FeatureUnion):
            self_features = self.transformer_list
        else:
            self_features = [(str(uuid.uuid4()), self)]

        if isinstance(other, FeatureUnion):
            other_features = other.transformer_list
        else:
            other_features = [(str(uuid.uuid4()), other)]

        return FeatureUnion(self_features + other_features)
