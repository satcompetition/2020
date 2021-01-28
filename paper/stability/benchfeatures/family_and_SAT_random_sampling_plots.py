#! /usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random

in_path = "par2scores.csv"
families = "benchmark_families.csv"
sat_answers = "instance_answers.csv"

def clean_features(data):
    mean = data.mean()
    idx = [i for i in range(data.shape[1]) if mean[i] != 10000]
    X_new = data.iloc[:,idx]
    return X_new

def read_csv_data(path):
    raw_data = pd.read_csv(path)
    data = raw_data.drop('solver', 1)
    solver = raw_data['solver']
    return solver, data

def calc_par2score(data2):
    value = data2.sum(axis=1)
    return value

def ranking(data1, score1):
    return [x for _,x in sorted(zip(score1, data1))]


def ranking_mat(ranks, solv):
    l = []
    for i in solv:
        l = l + [ranks.index(i)+1]
    return l

def spearman_vector(matrixr):
     cor = np.empty([len(matrixr)-1])

     for j in range(1,len(matrixr)):
         cor[j-1] = stats.spearmanr(matrixr[0], matrixr[j])[0]
     return cor

def random_sample(data, num):
    size = data.shape[1]
    indices = random.sample(range(size),num)
    subset = data.iloc[:,indices]
    return subset

def extract_sat_unsat(sat, data):
   if(sat == 0): sat_set = sat_answ.loc[sat_answ['answer'] == 'SAT-VERIFIED']
   else: sat_set = sat_answ.loc[sat_answ['answer'] == 'UNSAT']

   sat_bench = sat_set['benchmark'].str.strip()
   sat_bench = list("sat/"+sat_bench)
   data_bench=list(data.columns.values)

   intersect=list(set(sat_bench).intersection(data_bench))


   return data[intersect]

def get_sat_instances(sat, subset):
    if(sat == 0): sat_set = sat_answ.loc[sat_answ['answer'] == 'SAT-VERIFIED']
    else: sat_set = sat_answ.loc[sat_answ['answer'] == 'UNSAT']

    sat_bench = sat_set['benchmark'].str.strip()
    sat_bench = list("sat/"+sat_bench)
    data_bench=subset['benchmark_name'].str.strip()
    data_bench = list("sat/"+data_bench)
    intersect=list(set(sat_bench).intersection(data_bench))

    return(len(intersect))

def unknwn(data):
    sat_set = sat_answ.loc[sat_answ['answer'] == 'UNKNOWN']
    sat_bench = sat_set['benchmark'].str.strip()
    sat_bench = list("sat/"+sat_bench)
    data_bench=data['benchmark_name'].str.strip()
    data_bench = list("sat/"+data_bench)
    intersect=list(set(sat_bench).intersection(data_bench))
    return(len(intersect))

def fifty_fifty_sample_from_sat_unsat(data):
   sat_set = sat_answ.loc[sat_answ['answer'] == 'SAT-VERIFIED']
   unsat_set = sat_answ.loc[sat_answ['answer'] == 'UNSAT']

   satsize = sat_set.shape[0]
   unsatsize = unsat_set.shape[0]

   unsat_index = random.sample(range(unsatsize), unsatsize)
   unsat_set = unsat_set.iloc[unsat_index]
   unsat_set = unsat_set.replace('\n','', regex=True)
   unsat_bench = unsat_set['benchmark'].str.strip()
   unsat_bench = list("sat/"+unsat_bench)

   X_unsat = data[unsat_bench]
   mean = X_unsat.mean(axis=0)

   idx = [i for i in range(X_unsat.shape[1]) if mean[i] != 10000]
   X_unsat_new = X_unsat.iloc[:,idx]


   size = X_unsat_new.shape[1]

   sat_index = random.sample(range(satsize), size)

   sat_set = sat_set.iloc[sat_index]


   sat_set = sat_set.replace('\n','', regex=True)

   sat_bench = sat_set['benchmark'].str.strip()


   sat_bench = list("sat/"+sat_bench)


   X_sat = data[sat_bench]

   X_new = np.append(X_sat, X_unsat_new, axis = 1)


   return X_new

##############################################################################
def multiple_samples_fiftyfifty(data, runs, target, sc, solv):
   l1 = ranking_mat(target, solv)
   rankm = np.array([l1])
   plot_d = np.array([sc])
   corr=np.empty([runs-1])

   for i in range(1,runs):
     X_new = fifty_fifty_sample_from_sat_unsat(data)
     X_score = calc_par2score(X_new)
     rank = ranking(solv, X_score)
     l = ranking_mat(rank, solv)
     plot_d = np.append(plot_d, [X_score], axis = 0)
     corr[i-1] = stats.spearmanr(l1, l)[0]

   corr = np.around(corr, decimals=2)
   plotshort = plot_d[1:]
   corr_s = [x for x,_ in sorted(zip(corr, range(len(corr))), reverse=True)]
   range1 = [x for _,x in sorted(zip(corr, range(len(corr))), reverse=True)]
   plot_s = plotshort[range1]
   my_xticks = range(runs)
   fig = plt.gcf()
   fig.set_size_inches(11,8)

   plt.xticks(range(runs - 1),corr_s, rotation = 90)
   plt.plot(plot_s)
   plt.title('Rank changes for 20 runs for equally sampled SAT and UNSAT benchmarks',fontsize=18)
   plt.ylabel('Score',fontsize=16)
   plt.xlabel('Spearman\'s correlation for each run',fontsize=16)
   plt.grid(True)
   plt.savefig('plots/50-50-sat-unsat_random_sampling_scores.png')
   #plt.show()
   plt.clf()
   return corr

def multiple_samples_num(data, runs, target, sc, solv):
   l1 = ranking_mat(target, solv)
   rankm = np.array([l1])
   plot_d = np.array([sc])
   corr=np.empty([runs-1])

   for i in range(1,runs):
     X_new = random_sample(data, 158)
     X_score = calc_par2score(X_new)
     rank = ranking(solv, X_score)
     l = ranking_mat(rank, solv)
     plot_d = np.append(plot_d, [X_score], axis = 0)
     corr[i-1] = stats.spearmanr(l1, l)[0]

   corr = np.around(corr, decimals=2)
   plotshort = plot_d[1:]
   corr_s = [x for x,_ in sorted(zip(corr, range(len(corr))), reverse=True)]
   range1 = [x for _,x in sorted(zip(corr, range(len(corr))), reverse=True)]
   plot_s = plotshort[range1]
   my_xticks = range(runs)
   fig = plt.gcf()
   fig.set_size_inches(11,8)

   plt.xticks(range(runs-1),corr_s, rotation = 90)
   plt.plot(plot_s)
   plt.title('Rank changes for 20 runs for 230 random selected instances',fontsize=18)
   plt.ylabel('Score',fontsize=16)
   plt.xlabel('Spearman\'s correlation for each run',fontsize=16)
   plt.grid(True)
   plt.savefig('plots/230_instances_random_sampling_scores.png')
   #plt.show()
   plt.clf()
   return corr

def fam_leave_one_out(data, target, sc, solv):
    selected = []
    init = ranking_mat(target, solv)
    rankm = np.array([init])
    plot_d = np.array([sc])
    corr = []
    name = []
    trans = {
        "stedman-triples": "stedman-triples",
        "cnf-miter": "cnf-miter",
        "lam-discrete-geometry": "lam-discrete-geom.",
        "discrete-logarithm": "discrete-logarithm",
        "station-repacking": "station-repacking",
        "antibandwidth": "antibandwidth",
        "fermat": "fermat",
        "hgen": "hgen",
        "core-based-generator": "core-based-gen.",
        "vlsat": "vlsat",
        "baseball-lineup": "baseball-lineup",
        "schur-coloring": "schur-coloring",
        "cellular-automata": "cellular-automata",
        "bitvector": "bitvector",
        "cryptography": "cryptography",
        "edge-matching": "edge-matching",
        "cover": "cover",
        "ssp-0": "ssp-0",
        "hypertree-decomposition": "hypertree-decomp.",
        "coloring": "coloring",
        "polynomial-multiplication": "polynomial-multiply.",
        "tensors": "tensors",
        "termination": "termination",
        "tournament": "tournament",
        "influence-maximization": "influence-max.",
        "01-integer-programming": "01-integer-prog.",
        "timetable": "timetable",
    }
    for i in set(fam["family_name"]):

        subset = fam.loc[fam["family_name"] == i]

        subset = subset.replace("\n", "", regex=True)
        bench = subset["benchmark_name"].str.strip()
        bench = list("sat/" + bench)
        X_new = data.drop(bench, 1)

        X_score = calc_par2score(X_new)
        rank = ranking(solv, X_score)
        l = ranking_mat(rank, solv)
        rankm = np.append(rankm, [l], axis=0)
        plot_d = np.append(plot_d, [X_score], axis=0)

        unkn = unknwn(subset)
        solved = subset.shape[0] - unkn

        name = name + [trans[i] + " (" + str(solved) + "/" + str(subset.shape[0]) + ")"]

    corr = spearman_vector(rankm)

    name_s = [x for _, x in sorted(zip(corr, name))]
    corr_s = [x for x, _ in sorted(zip(corr, name))]
    name_s = [x.replace("final_", "") for x in name_s]
    fig = plt.gcf()
    # fig.set_size_inches(11,8)
    plt.gcf().subplots_adjust(bottom= 0.46, top= 0.95)
    plt.xticks(range(len(name_s)), name_s, rotation=90)
    plt.xlabel(
        "Family name (number of solved instances / number of instances in the family)"
    )
    plt.ylabel("Spearman's correlation")
    plt.plot(corr_s)
    plt.grid(True)

    # plt.title("Change of rank correlation \n under removal of a certain benchmark family")
    plt.savefig("plots/fam_leave_one_out_corr.png")
    # plt.show()
    plt.clf()

    return corr_s

def fam_leave_one_out_unsat(data, target, sc, solv):
   selected = []
   init = ranking_mat(target, solv)
   rankm = np.array([init])
   plot_d = np.array([sc])
   corr = []
   name = []

   for i in set(fam['family_name']):

       subset = fam.loc[fam['family_name'] == i]

       subset = subset.replace('\n','', regex=True)
       bench = subset['benchmark_name'].str.strip()
       bench = list("sat/"+bench)


       X_new = data.drop(bench, 1)


       X_new = extract_sat_unsat(1, X_new)

       X_score = calc_par2score(X_new)
       rank = ranking(solv, X_score)
       l = ranking_mat(rank, solv)
       rankm = np.append(rankm, [l], axis = 0)
       plot_d = np.append(plot_d, [X_score], axis = 0)
       size=get_sat_instances(1,subset)
       name = name + [i+" ("+str(size)+")"]


   corr = spearman_vector(rankm)

   name_s = [x for _,x in sorted(zip(corr, name))]
   corr_s = [x for x,_ in sorted(zip(corr, name))]
   name_s = [x.replace("final_","") for x in name_s]
   fig = plt.gcf()
   fig.set_size_inches(11,8)
   plt.gcf().subplots_adjust(bottom=0.35)
   plt.xticks(range(len(name_s)),name_s, rotation = 90)
   plt.xlabel("Family name (number of unsat instances in the family)")
   plt.ylabel("Spearman\'s correlation")
   plt.plot(corr_s)
   plt.grid(True)

   plt.title("Change of rank correlation \n under removal of a certain benchmark family considering only unsat instances")
   plt.savefig("plots/fam_leave_one_out_corr_unsat.png")
   #plt.show()
   plt.clf()

   return corr_s

def fam_leave_one_out_sat(data, target, sc, solv):
   selected = []
   init = ranking_mat(target, solv)
   rankm = np.array([init])
   plot_d = np.array([sc])
   corr = []
   name = []

   for i in set(fam['family_name']):

       subset = fam.loc[fam['family_name'] == i]

       subset = subset.replace('\n','', regex=True)
       bench = subset['benchmark_name'].str.strip()
       bench = list("sat/"+bench)


       X_new = data.drop(bench, 1)


       X_new = extract_sat_unsat(0, X_new)

       X_score = calc_par2score(X_new)
       rank = ranking(solv, X_score)
       l = ranking_mat(rank, solv)
       rankm = np.append(rankm, [l], axis = 0)
       plot_d = np.append(plot_d, [X_score], axis = 0)
       size=get_sat_instances(0,subset)
       name = name + [i+" ("+str(size)+")"]


   corr = spearman_vector(rankm)

   name_s = [x for _,x in sorted(zip(corr, name))]
   corr_s = [x for x,_ in sorted(zip(corr, name))]
   name_s = [x.replace("final_","") for x in name_s]
   fig = plt.gcf()
   fig.set_size_inches(11,8)
   plt.gcf().subplots_adjust(bottom=0.35)
   plt.xticks(range(len(name_s)),name_s, rotation = 90)
   plt.xlabel("Family name (number of sat instances in the family)")
   plt.ylabel("Spearman\'s correlation")
   plt.plot(corr_s)
   plt.grid(True)

   plt.title("Change of rank correlation \n under removal of a certain benchmark family considering only sat instances")
   plt.savefig("plots/fam_leave_one_out_corr_sat.png")
   #plt.show()
   plt.clf()

   return corr_s

##########################################################################################
fam = pd.read_csv(families)
sat_answ = pd.read_csv(sat_answers)

solver, X = read_csv_data(in_path)
X_filtered = clean_features(X)

score = calc_par2score(X)
comp_results = ranking(solver, score)

runs = 30
plot = 0

X_sat=extract_sat_unsat(0, X)
scoresat = calc_par2score(X_sat)
comp_resultssat = ranking(solver, scoresat)

X_unsat=extract_sat_unsat(1, X)
scoreunsat = calc_par2score(X_unsat)
comp_resultsunsat = ranking(solver, scoreunsat)

# Plot family leave one out
fam_leave_one_out(X, comp_results, score, solver)
# fam_leave_one_out_sat(X, comp_resultssat, scoresat, solver)
# fam_leave_one_out_unsat(X, comp_resultsunsat, scoreunsat, solver)

# # PLOT 50-50
# multiple_samples_fiftyfifty(X, runs, comp_results, score, solver)

# # PLOT 230 selected benchmarks
# multiple_samples_num(X_filtered, runs, comp_results, score, solver)
