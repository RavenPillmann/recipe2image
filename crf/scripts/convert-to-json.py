#!/usr/bin/env python
from __future__ import print_function

import sys
import json

import crf_utils

if len(sys.argv) < 2:
    sys.stderr.write('Usage: make-data.py FILENAME')
    sys.exit(1)

print(json.dumps(crf_utils.import_data(open(sys.argv[1])), indent=4))
