#!/bin/bash
python -m build
python -m pip install dist/*.tar.gz
