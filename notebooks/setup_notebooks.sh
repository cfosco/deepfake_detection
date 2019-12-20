#!/bin/bash

# Start from directory of script
cd "$(dirname "${BASH_SOURCE[0]}")"

# Set up symlinks for the example notebooks
ln -s ../logs .
ln -s ../data .
ln -s ../models .
ln -s ../weights .
