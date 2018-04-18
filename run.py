#!/usr/bin/python
import argparse
import csv
import os
import subprocess

BINARY_NAME = "./triangle"
BECNHMARK_TYPE_SUFFIX = "_adj"

local_benchmarks = ["amazon0302"]

parser = argparse.ArgumentParser(description='Runner for triangle benchmarks')
parser.add_argument('-i', '--input', help = "Input CSV file", dest = 'csv_file', required = True)
parser.add_argument('-b', '--benchmark-dir', help = "benchmarks directory", dest = 'benchmark_dir', required = True)

def main():
    args = parser.parse_args()

    if os.path.isfile(BINARY_NAME) == False:
        print BINARY_NAME + " binary not found!"
        return

    with open(args.csv_file, mode="r" ) as file_in:
        benchmarks = csv.DictReader(file_in)
        for benchmark in benchmarks:
            num_nodes = benchmark['num_nodes']
            num_edges = benchmark['num_edges']
            dataset = args.benchmark_dir + benchmark['dataset'] + BECNHMARK_TYPE_SUFFIX

            if benchmark['dataset'] in local_benchmarks:
                print "Running " + benchmark['dataset']
                subprocess.call([BINARY_NAME, dataset, num_nodes, num_edges])
            

if __name__ == "__main__":
    main()
