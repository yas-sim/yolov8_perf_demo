#!/usr/bin/env /usr/bin/bash

source /venv/bin/activate
mkdir -p /work/share
cd /work/share
python3 /work/conv.py
deactivate
