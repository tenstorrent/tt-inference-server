#!/bin/bash

for i in {1..100}; do
    echo "Running iteration $i of 100"
    python3 run.py --model Llama-3.3-70B-Instruct --device galaxy --workflow release
    echo "Completed iteration $i"
    echo "---"
done

echo "All 100 iterations completed!"

