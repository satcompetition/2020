#! /usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv_data(path):
    raw_data = pd.read_csv(path)
    raw_data.rename(columns=lambda x: x[4:] if str(x).startswith('sat/') else x, inplace=True)
    return raw_data

def remove_unsolved_instances(df,penalty):
    data = df.copy()
    unsolved_instances = []
    for col in data.columns[1:]: #columns[0] == 'solvers'
        if data[col].eq(penalty).all(): # Penalty running time means that the instance was not solved, i.e. TO or MO happened during solving
            unsolved_instances.append(col)

    data.drop(unsolved_instances, axis=1, inplace=True)
    print('Eliminating %d unsolved instances from the data.' %len(unsolved_instances))
    return data

def introduce_par2score_column(df):
    data = df.copy()
    data.insert(loc=len(data.columns),column='par2_score',value=data.iloc[:,:].sum(axis=1,numeric_only=True))
    return data

def sort_by_score(scores,solvers):
    solver_score_pairs = zip(scores,solvers)
    return sorted(solver_score_pairs)

def get_position(solver,rank):
    for (idx,r) in zip(range(len(rank)),rank):
        (rank_score,rank_solver) = r
        if rank_solver == solver:
            return idx+1
    assert False

def spearman_correlation(rank1,rank2): #alternative: scipy.starts.spearmanr
    assert (len(rank1) == len(rank2))
    diff_square_sum = 0
    for _,solver in rank1:
        pos1 = get_position(solver,rank1)
        pos2 = get_position(solver,rank2)
        diff_square_sum += ((pos1-pos2)**2)
    n = len(rank1)
    nom = 6*diff_square_sum
    dnom = n*((n**2)-1)
    spc = 1. - nom/dnom

    return spc

def select_random_features(amount,times,feature_list):
    assert amount <= len(feature_list)
    selected_features = []
    for t in range(times):
        selected_features.append(np.random.choice(feature_list,amount,replace=False)) #random sampling without replacement - sample WITH replacement: random.choices

    return selected_features


def random_sampling(data_frame,original_rank,original_features,solver_names,repeat,ub,step):
    diffs = np.zeros((len(np.arange(0,ub,step)),repeat),dtype=int)
    df = data_frame.loc[:,data_frame.columns.str.endswith('.bz2')]
    removed2corr_mean = {}
    removed2corr_std = {}
    for amount in np.arange(0,ub,step):
        features_to_remove = select_random_features(amount,repeat,original_features)
        correlations = []
        for (idx,to_remove) in zip(range(len(features_to_remove)),features_to_remove):
            filtered_data = df.loc[:,df.columns.difference(to_remove)]
            filtered_data = introduce_par2score_column(filtered_data)
            filtered_scores = round(filtered_data['par2_score'],3)
            filtered_rank = sort_by_score(filtered_scores,solver_names)
            correlations.append(spearman_correlation(original_rank,filtered_rank))
        removed2corr_mean[amount] = np.mean(correlations)
        removed2corr_std[amount] = np.std(correlations)
    return removed2corr_mean,removed2corr_std


def random_sampling_plot(data_frame,original_ranks,original_instances,solver_names,title,save_path=None):
    sampling_freq = 50
    features_to_remove_max = len(original_instances)
    step = 1
    corr_means,corr_stds = random_sampling(data_frame,original_ranks,original_instances,solver_names,sampling_freq,features_to_remove_max,step)
    sorted_means = sorted(corr_means.items()) # sorted by key, return a list of tuples

    number_of_removed, correlation_mean = zip(*sorted_means)
    sorted_stds = sorted(corr_stds.items())
    number_of_removed,correlation_std = zip(*sorted_stds)

    plt.figure()
    fig = plt.figure()
    # fig.set_size_inches(9.6, 6.8)
    # plt.title(title,fontsize=18)
    plt.yticks(np.arange(0.,1.09,0.1))
    # plt.ylabel('Spearman\'s correlation',fontsize=16)
    # plt.xlabel('Number of randomly removed instances',fontsize=16)
    plt.ylabel('Spearman\'s correlation')
    plt.xlabel('Number of randomly removed instances')
    plt.errorbar(number_of_removed, correlation_mean, correlation_std, linestyle='None', marker='^',elinewidth=1)
    plt.hlines([0.8,0.99,0.6,0.4,0.2],0,max(number_of_removed)+5,linestyle='dashed',color='m')
    plt.grid(True)

    print(title)
    hist,bins = np.histogram(correlation_mean, bins=[0.,0.2,0.4,0.6,0.8,0.99,1.])
    bin_counts = zip(bins,bins[1:],hist)
    for bin_start, bin_end, count in list(bin_counts):
        print('%.2f-%.2f: %d (%.2f %%)' % (bin_start, bin_end, count, (count/len(correlation_mean))*100))

    if save_path:
        plt.savefig(save_path)
    #plt.show()


np.random.seed(15487061) #Figures of paper are generated with this seed
penalty = 10000
in_path = "par2scores.csv"
data = read_csv_data(in_path)
data = remove_unsolved_instances(data,penalty)
data = introduce_par2score_column(data)

sat_answers = "instance_answers.csv"
answer_df = pd.read_csv(sat_answers)
sat_column_names = ['solver']+list(answer_df.loc[answer_df['answer'] == 'SAT-VERIFIED']['benchmark'])
unsat_column_names =  ['solver']+list(answer_df.loc[answer_df['answer'] == 'UNSAT']['benchmark'])

sat_data = data.loc[:,data.columns.intersection(sat_column_names)]
unsat_data = data.loc[:,data.columns.intersection(unsat_column_names)]

solver_names = data['solver']
original_scores = round(data['par2_score'],3)
original_rank = sort_by_score(original_scores,solver_names)
original_instances = list(data.loc[:,data.columns.str.endswith('.bz2')])

sat_data = introduce_par2score_column(sat_data)
sat_original_scores = round(sat_data['par2_score'],3)
sat_original_rank = sort_by_score(sat_original_scores,solver_names)
sat_original_instances = list(sat_data.loc[:,sat_data.columns.str.endswith('.bz2')])

unsat_data = introduce_par2score_column(unsat_data)
unsat_original_scores = round(unsat_data['par2_score'],3)
unsat_original_rank = sort_by_score(unsat_original_scores,solver_names)
unsat_original_instances = list(unsat_data.loc[:,unsat_data.columns.str.endswith('.bz2')])
print('Found %d SAT and %d UNSAT problems from %d problems.' % (len(sat_original_instances),len(unsat_original_instances),len(original_instances)))

title = 'Mean and standard deviation of\n ranking correlations at random problem instance removals\n(repeated 50x each simple random sampling)'
random_sampling_plot(data,original_rank,original_instances,solver_names,title,'plots/ALL_random_smpling_correlations.png')
# title = 'Mean and standard deviation of ranking correlations\n at random problem instance removals considering only SAT instances\n(repeated 50x each simple random sampling)'
# random_sampling_plot(sat_data,sat_original_rank,sat_original_instances,solver_names,title,'plots/SAT_randomSampling_correlations.png')
# title = 'Mean and standard deviation of ranking correlations\n at random problem instance removals considering only UNSAT instances\n(repeated 50x each simple random sampling)'
# random_sampling_plot(unsat_data,unsat_original_rank,unsat_original_instances,solver_names,title,'plots/UNSAT_randomSampling_correlations.png')
