#! /usr/bin/env python3
import csv
import pandas as pd

penalty = 10000
verifier_failed_markers = ['DRATmemout','DRATtimeout','TIMEOUT','RATcfoapp','Enc','ccbnd']
solver2scores = {}
header = True
benchmark_set = set()

answers = {}

with open('main.csv','r') as main_input:
    for line in main_input:
        if header:
            header = False
            continue

        cells = line.strip().split(',')
        benchmark = cells[0]
        benchmark_set.add(benchmark)
        solver = cells[1] + '-' + cells[2] #solver-configuration
        if benchmark[4:] not in answers: #remove prefix 'sat/'
            answers[benchmark[4:]] = []
        answers[benchmark[4:]].append(cells[5])
        if cells[4] != 'complete' or cells[7] in verifier_failed_markers:
            score = penalty
        else:
            score = cells[3]

        if solver in solver2scores:
            solver2scores[solver][benchmark] = score
        else:
            solver2scores[solver] = {}
            solver2scores[solver]['solver'] = solver
            solver2scores[solver][benchmark] = score

#sorting is not relevant to the plots, it is here just to reproduce the used data-files exactly
output_header = ['solver'] + sorted(list(benchmark_set), key=lambda s: s.lower().replace('_','!'))
sorted_solvers_list = sorted(solver2scores.keys(), key=lambda s: s.lower().replace('+','.').replace('_','!'))

output_file = 'par2scores.csv'
try:
    with open(output_file,'w') as output_csv:
        for instance in output_header[:-1]:
            output_csv.write(instance + ',')
        output_csv.write(output_header[-1]+'\n')
        for solver in sorted_solvers_list:
            output_csv.write(solver+',')
            for instance in output_header[1:-1]:
                output_csv.write(str(solver2scores[solver][instance])+',')
            output_csv.write(str(solver2scores[solver][output_header[-1]])+'\n')
except IOError:
    print("I/O error")


output_file = 'instance_answers.csv'
try:
    with open(output_file,'w') as output_csv:
        output_csv.write('benchmark,answer\n')
        for instance,answer in sorted(answers.items()):
            answer_set = set(answer)
            assert  not (('SAT-VERIFIED' in answer_set) and ('UNSAT' in answer_set))
            if 'UNKNOWN' in answer_set and len(answer_set) > 1:
                answer_set.remove('UNKNOWN')
            assert len(answer_set) == 1
            if instance == 'sted1_0x1e3-100.cnf.bz2' and not('SAT-VERIFIED' in answer_set):
                # This instance was solved by smallsat but remained UNKNOWN in main.csv
                # Independent run showed that the problem instance is actually SAT
                output_csv.write(instance + ',SAT-VERIFIED\n')
            else:
                output_csv.write(instance + ',' + answer_set.pop()+'\n')
except IOError:
    print("I/O error")
