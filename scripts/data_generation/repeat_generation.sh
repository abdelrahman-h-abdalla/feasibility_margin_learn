#!/bin/bash
# This script expects one argument: stance_feet_str
stance_feet_str=$1

for run in {1..100}; do
	python stability_margin.py "$stance_feet_str";
done
