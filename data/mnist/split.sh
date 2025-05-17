#!/bin/bash
# split.sh

python3 prepare.py -t input_abs.txt --method char
outdir="../mnist_abs"
mkdir "$outdir"
mv *.pkl "$outdir"
mv *.bin "$outdir"

python3 prepare.py -t input_column.txt --method char
outdir="../mnist_column"
mkdir "$outdir"
mv *.pkl "$outdir"
mv *.bin "$outdir"

python3 prepare.py -t input_row.txt --method char
outdir="../mnist_row"
mkdir "$outdir"
mv *.pkl "$outdir"
mv *.bin "$outdir"

python3 prepare.py -t input.txt --method char
