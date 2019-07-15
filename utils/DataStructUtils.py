#! /usr/bin/env python
# coding: utf-8
import os, sys
from collections import OrderedDict

def sort_orddict(d):
    d1 = sorted(d.items(), key=lambda item: item[0])
    return OrderedDict(d1)

def add_and_filte_orddict(d, k, v, l):
    d1 = sorted(d.items(), key=lambda item: item[0])
    d2 = d1
    d3 = []
    if len(d1) > l:
        d2 = OrderedDict(d1[0:l])
        d3 = OrderedDict(d1[l:])
    return OrderedDict(d2), OrderedDict(d3)