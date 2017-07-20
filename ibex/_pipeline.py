from __future__ import absolute_import

from sklearn import pipeline

from ._frame_mixin import FrameMixin


class Pipeline(pipeline.Pipeline):
    def __getitem__(self, ind):
        return self.steps[ind][1]


def make_pipeline(*steps):
    orig = pipeline.make_pipeline(*steps)
    return Pipeline(orig.steps)
