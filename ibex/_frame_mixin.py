from __future__ import absolute_import

import operator

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
            self.set_params(columns=columns)

    @property
    def x_columns(self):
        return self.__cols

    @x_columns.setter
    def x_columns(self, columns):
        try:
            self.__cols
        except AttributeError:
            self.__cols = columns

        if set(columns) != set(self.__cols):
            raise KeyError()

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
        from ._pipeline import Pipeline

        if issubclass(type(other), pipeline.Pipeline):
            others = [operator.itemgetter(1)(e) for e in other.steps]
        else:
            others = [other]
        others = [self] + others

        import random
        import string
        combined = [(''.join(random.choice(string.digits) for _ in range(10)), o) for o in others]

        return Pipeline(combined)

    def __ror__(self, other):
        from ._pipeline import Pipeline

        if issubclass(type(other), pipeline.Pipeline):
            others = [operator.itemgetter(1)(e) for e in other.steps]
        else:
            others = [other]
        others += [self]

        combined = [(type(o).__name__, o) for o in others]
        import random
        import string
        combined = [(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)), o) for o in others]

        return Pipeline(combined)

    def __add__(self, other):
        from ._feature_union import FeatureUnion

        if isinstance(self, FeatureUnion):
            self_features = [operator.itemgetter(1)(e) for e in self.transformer_list]
        else:
            self_features = [self]

        if isinstance(other, FeatureUnion):
            other_features = [operator.itemgetter(1)(e) for e in other.transformer_list]
        else:
            other_features = [other]

        import random
        import string
        combined = [(''.join(random.choice(string.digits) for _ in range(10)), o) for o in self_features + other_features]

        return FeatureUnion(combined)
