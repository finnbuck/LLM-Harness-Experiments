# This folder is for storing different network types.
# Scripts for generating networks should also live here.

SELF_LOOP = {1 : [1]}

PAIR = {1 : [2], 2 : [1]}

TRIANGLE = {1 : [2], 2 : [3], 3 : [1]}

SQUARE = {1 : [2, 3], 2 : [1, 4], 3 : [1, 4], 4 : [2, 3]}


