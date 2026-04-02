"""
tasks/__init__.py
=================
Task base class and registry.
"""

from tasks.easy_task import EasyTask
from tasks.medium_task import MediumTask
from tasks.hard_task import HardTask

__all__ = ["EasyTask", "MediumTask", "HardTask"]
