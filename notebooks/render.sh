#!/bin/bash
set -e
echo "Using template $1"

jupytext --to py timing-example.ipynb
cp timing-example.ipynb timing-example-copy.ipynb
python control_tags.py timing-example-copy.ipynb

jupyter-nbconvert --to html --template $1 --output exper_block.html timing-example-copy.ipynb
