from __future__ import absolute_import

import operator
import collections

from sklearn import pipeline


def _make_pipeline_steps(objs):
    names = [type(o).__name__.lower() for o in objs]
    name_counts = collections.Counter(names)
    unique_names = []
    for name in names:
        if name_counts[name] > 1:
            unique_names.append(name + '_' + str(name_counts[name] - 1))
            name_counts[name] -= 1
        else:
            unique_names.append(name)

    return list(zip(unique_names, objs))


class FrameMixin(object):
    """
    A base class for steps taking pandas entities, not numpy entities.

    Subclass this step to indicate that a step takes pandas entities.

    Example:

        >>> import ibex
        >>>
        >>> class GroupbyAggregator(ibex.FrameMixin):
        ...     pass
    """

    @property
    def x_columns(self):
        """
        The columns set in the last call to fit.

        Set this property at fit, and call it in other methods:

        """
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
        """
        Pipes the result of this stage to other.


        Arguments:
            other: A different step object whose class subclasses this one.

        Returns:
            :py:class:`sklearn.pipeline.Pipeline`
        """

        if issubclass(type(other), pipeline.Pipeline):
            others = [operator.itemgetter(1)(e) for e in other.steps]
        else:
            others = [other]
        combined = [self] + others

        return pipeline.Pipeline(_make_pipeline_steps(combined))

    def __ror__(self, other):
        """

        Returns:
            :py:class:`sklearn.pipeline.Pipeline`
        """

        if issubclass(type(other), pipeline.Pipeline):
            others = [operator.itemgetter(1)(e) for e in other.steps]
        else:
            others = [other]
        combined = others + [self]

        return pipeline.Pipeline(_make_pipeline_steps(combined))

    def __add__(self, other):
        """

        Returns:
            :py:class:`ibex.sklearn.pipeline.FeatureUnion`
        """

        from ._feature_union import _FeatureUnion

        if isinstance(self, _FeatureUnion):
            self_features = [operator.itemgetter(1)(e) for e in self.transformer_list]
        else:
            self_features = [self]

        if isinstance(other, _FeatureUnion):
            other_features = [operator.itemgetter(1)(e) for e in other.transformer_list]
        else:
            other_features = [other]

        combined = self_features + other_features

        return _FeatureUnion(_make_pipeline_steps(combined))
