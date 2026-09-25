# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:15:54 2026

@author: Nicola
"""

from abc import ABC, abstractmethod, abstractproperty
from datetime import datetime


class DayCountConvention(ABC):

    @abstractmethod
    def year_fraction(self, start, end):
        pass


class Actual365(DayCountConvention):

    def year_fraction(self, start, end):
        return (end - start).days / 365.0


class Actual360(DayCountConvention):

    def year_fraction(self, start, end):
        return (end - start).days / 360.0
    

class Actual252(DayCountConvention):
    
    def year_fraction(self, start, end):
        
        return (end-start).days / 252