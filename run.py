#!/usr/bin/python
import argparse
import csv
import os
import subprocess

BINARY_NAME = "./triangle-omp"
BECNHMARK_TYPE_SUFFIX = "_adj"

local_benchmarks = [
"oregon1_010526"
]

parser = argparse.ArgumentParser(description='Runner for triangle benchmarks')
parser.add_argument('-i', '--input', help = "Input CSV file", dest = 'csv_file', required = True)
parser.add_argument('-b', '--benchmark-dir', help = "benchmarks directory", dest = 'benchmark_dir', required = True)
parser.add_argument('-m', '--mode', help = "Mode auto/manual", dest = 'mode', required = True)

total_count = 0

def run_auto(benchmark, num_nodes, num_edges, dataset):
    global total_count
    if os.path.isfile(dataset + "_IA.txt") and os.path.isfile(dataset + "_JA.txt"):
        print "Running " + benchmark['dataset']
        subprocess.call([BINARY_NAME, dataset, num_nodes, num_edges])
        total_count += 1

def run_manual(benchmark, num_nodes, num_edges, dataset):
    global total_count
    if benchmark['dataset'] in local_benchmarks:
        print "Running " + benchmark['dataset']
        subprocess.call([BINARY_NAME, dataset, num_nodes, num_edges])
        total_count += 1

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

            if args.mode == "manual":
                run_manual(benchmark, num_nodes, num_edges, dataset)
            else:
                run_auto(benchmark, num_nodes, num_edges, dataset)

    print "Total benchmarks = ", total_count
                
            

if __name__ == "__main__":
    main()
