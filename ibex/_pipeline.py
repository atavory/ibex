from __future__ import absolute_import

from sklearn import pipeline

from ._frame_mixin import FrameMixin


class Pipeline(pipeline.Pipeline):
    pass


def make_pipeline(*steps):
    orig = pipeline.make_pipeline(*steps)
    params = orig.get_params()
    return Pipeline(params['steps'])
