#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

from numpy import ndarray, eye, zeros, copy as npcopy
from spatialmath.base import r2q
from abc import ABC

# from roboticstoolbox.robot.ETS import ETS
from typing import Type

from spatialgeometry.SceneNode import SceneNode


class SceneGroup(SceneNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def hello(self):
        print("HEllo")
