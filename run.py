import argparse
import TabularDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpi4py import MPI
from train_blackbox import train_clf
from train_fair_blackbox import train_clf_fair
from metric import demographic_parity_diff, equalized_odds_diff
from ExponentiatedGradientwrapper import ExpoGrad
from train_explaination import Metric

# Setting up argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--seeds', nargs='+', type=int, help='List of seed values for explanations', default=[0])
parser.add_argument('--dataset', choices=list(TabularDataset.dataset_params.keys()), type=str, required=True)
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--train_blackbox_model', action='store_true', help='Train BB model')
parser.add_argument('--model_name', choices=['xgb', 'rf'], type=str, required=True, help='Name of the model to train')
parser.add_argument('--scoring', default='accuracy', type=str, help='Scoring method for the model')
parser.add_argument('--reductionist_type', type=str, choices=['DP', 'EO'])
parser.add_argument('--reductionist_difference_bound', nargs='*', type=float, 
                    default=np.arange(0.01, 1, 0.01).tolist(), help='List of reductionist difference bounds to iterate over (optional)')
parser.add_argument('--train_fair_model', action='store_true', help='Train fair model')
parser.add_argument('--adjusted_bound', type=float, default=None, help='Adjusted bound for fair training')
parser.add_argument('--plot_metrics', action='store_true', help='Plot fairness metrics')
parser.add_argument('--per_values', nargs='+', type=float, help='List of fairness constraint thresholds for plotting')
parser.add_argument('--dem_parity_values', nargs='+', type=float, help='List of Demographic Parity Differences for plotting')
parser.add_argument('--eq_odds_values', nargs='+', type=float, help='List of Equalized Odds Differences for plotting')
parser.add_argument('--run_max_fidelity', action='store_true', help='Run max fidelity gap metrics')
parser.add_argument('--run_stab', action='store_true', help='Run average stability metrics')
parser.add_argument('--run_reco', action='store_true', help='Run consistency metrics')
parser.add_argument('--gender', type=str, choices=['male', 'female', 'both'], help='Gender for which to run the metrics')
parser.add_argument('--age', type=str, choices=['g0', 'g1', 'both'], help='Group for each range where g_0 corresponds to age leq 43')
parser.add_argument('--race', type=str, choices=['afr', 'cau', 'both'], help='Race for which to run the metrics')

args = parser.parse_args()

# Load dataset
dataset = TabularDataset.Dataset(args.dataset)
X_train_df, X_train, X_test_df, X_test, X_explain_df, X_explain, y_train, y_test, y_explain, s_train, s_test, s_explain = dataset.get_data(seed=args.seed)

def run_training(args):
    """Runs the training script with the given arguments."""
    unfair_clf, predictions_test, proba_test, test_accuracy = train_clf(args.model_name, 
                                                                         X_train, 
                                                                         y_train, 
                                                                         X_test, 
                                                                         y_test, 
                                                                         scoring=args.scoring)
    return unfair_clf, predictions_test

def run_fair_training(args, bound):
    """Runs the fair training script with the given arguments."""
    fair_clf, un_mod, predictions_test, test_accuracy = train_clf_fair(args.model_name, 
                                                                      X_train, 
                                                                      y_train, 
                                                                      X_test, 
                                                                      y_test, 
                                                                      s_train,
                                                                      args.reductionist_type, 
                                                                      adjusted_bound=bound)
    return fair_clf, un_mod, predictions_test

def calculate_fairness_metrics(pred, y_test, s_test, reductionist_type):
    """Calculates the fairness metrics for the predictions based on the specified reductionist type."""
    if reductionist_type == 'DP':
        unfairness = demographic_parity_diff(y_test, pred, s_test)
    elif reductionist_type == 'EO':
        unfairness = equalized_odds_diff(y_test, pred, s_test)
    else:
        raise ValueError(f"Unknown reductionist type: {reductionist_type}")
    
    return unfairness


def run_parallel_bound_analysis(args, bound, base_unfairness, seeds, output_path):
    all_max_fidelity_results, all_stab_results, all_reco_results = [], [], []
    fairness_results = []
    dem_parity_values = []

    adj_bound = bound * base_unfairness
    fair_clf, un_mod, fair_predictions = run_fair_training(args, adj_bound)
    dem_parity = calculate_fairness_metrics(fair_predictions, y_test, s_test, args.reductionist_type)

    for rseed in seeds:
        np.random.seed(rseed)
        _, _, _, _, X_explain_df, X_explain, _, _, y_explain, _, _, _ = dataset.get_data(seed=rseed)
        
        wrapped_model = ExpoGrad(fair_clf)

        # Determine sensitive attribute based on dataset
        if args.dataset == 'ACSIncome':
            sensitive_attribute = 'SEX'
            k_top_features = 20
            samples, labels = select_samples_by_gender(X_explain, X_explain_df, y_explain, sensitive_attribute, args.gender)
        
        elif args.dataset == 'ACSEmployment':
            sensitive_attribute = 'AGEP'
            k_top_features = 24
            samples, labels = select_samples_by_age(X_explain, X_explain_df, y_explain, sensitive_attribute, args.age)
        
        elif args.dataset == 'COMPAS':
            sensitive_attribute = 'race'
            k_top_features = 4
            samples, labels = select_samples_by_race(X_explain, X_explain_df, y_explain, sensitive_attribute, args.race)
        
        else:
            raise ValueError(f'No dataset with name {args.dataset}')

        metric_eval = Metric(wrapped_model)

        if args.run_max_fidelity:            
            max_fidelity_results = metric_eval.max_fidelity_gap_from_average(samples, labels, df=X_explain_df,
                                                                             sensitive_feature=X_explain_df[[sensitive_attribute]].columns[0], 
                                                                             k_top_features=k_top_features)
            all_max_fidelity_results.append(max_fidelity_results)

        if args.run_stab:
            stab_results = metric_eval.stab(samples, labels)
            all_stab_results.append(stab_results)

        if args.run_reco:
            reco_results = metric_eval.reco(samples, labels)
            all_reco_results.append(reco_results)

        # Calculate and store demographic parity for each seed
        dem_parity = calculate_fairness_metrics(fair_predictions, y_test, s_test, args.reductionist_type)
        dem_parity_values.append(dem_parity)
        
    # Compute mean demographic parity over all seeds
    mean_dem_parity = np.mean(dem_parity_values)
    fairness_results.append({'bound': bound, 'demographic_parity': mean_dem_parity})

    fairness_df = pd.DataFrame(fairness_results)

    # Save the results to CSV
    fairness_filename = os.path.join(output_path, f'fairness_results_{args.model_name}_{args.dataset}.csv')
    fairness_df.to_csv(fairness_filename, index=False, mode='a', header=not os.path.exists(fairness_filename))

    return mean_dem_parity, all_max_fidelity_results, all_stab_results, all_reco_results

def select_samples_by_gender(X_explain, X_explain_df, y_explain, sensitive_attribute, gender):
    """Select samples based on gender."""
    X_explain_df = X_explain_df.reset_index(drop=True)
    y_explain = y_explain.reset_index(drop=True)
    if gender == 'female':
        samples = X_explain[X_explain_df[sensitive_attribute] == 1]
        labels = y_explain[X_explain_df[sensitive_attribute] == 1]
    elif gender == 'male':
        samples = X_explain[X_explain_df[sensitive_attribute] == 0]
        labels = y_explain[X_explain_df[sensitive_attribute] == 0]
    else: 
        samples = X_explain
        labels = y_explain
    return samples, labels

def select_samples_by_age(X_explain, X_explain_df, y_explain, sensitive_attribute, age):
    """Select samples based on age group."""
    X_explain_df = X_explain_df.reset_index(drop=True)
    y_explain = y_explain.reset_index(drop=True)
    if age == 'g1':
        samples = X_explain[X_explain_df[sensitive_attribute] == 1]
        labels = y_explain[X_explain_df[sensitive_attribute] == 1]
    elif age == 'g0':
        samples = X_explain[X_explain_df[sensitive_attribute] == 0]
        labels = y_explain[X_explain_df[sensitive_attribute] == 0]
    else:  
        samples = X_explain
        labels = y_explain
    return samples, labels

def select_samples_by_race(X_explain, X_explain_df, y_explain, sensitive_attribute, race):
    """Select samples based on race."""
    X_explain_df = X_explain_df.reset_index(drop=True)
    
    # If y_explain is a NumPy array, we don't need to reset index
    if isinstance(y_explain, pd.Series) or isinstance(y_explain, pd.DataFrame):
        y_explain = y_explain.reset_index(drop=True)
    
    if race == 'cau':
        samples = X_explain[X_explain_df[sensitive_attribute] == 1]
        labels = y_explain[X_explain_df[sensitive_attribute] == 1]
    elif race == 'afr':
        samples = X_explain[X_explain_df[sensitive_attribute] == 0]
        labels = y_explain[X_explain_df[sensitive_attribute] == 0]
    else:  # both
        samples = X_explain
        labels = y_explain
        
    return samples, labels


def split(container, count):
    """Splits the container into 'count' parts."""
    return [container[i::count] for i in range(count)]

def process_and_save_results(combined_results, args, output_path, graph_path):
    """Processes and saves results from the analysis."""
    if not os.path.isdir(output_path):
        raise ValueError(f"The specified path '{output_path}' is not a directory.")
    
    os.makedirs(output_path, exist_ok=True)

    dem_parity_values = []
    reductionist_bounds = args.reductionist_difference_bound

    for bound in reductionist_bounds:
        result = combined_results[bound]
        dem_parity_values.append(result['dem_parity'])

        # Save fidelity and stability DataFrames to CSV
        if args.run_max_fidelity and result['max_fidelity'] is not None:
            sensitive_attribute = args.gender or args.age or args.race
            max_fidelity_filename = os.path.join(output_path, f'max_fidelity_{sensitive_attribute}_{bound}_{args.dataset}_{args.model_name}.csv')
            combined_max_fidelity_df = pd.concat(result['max_fidelity'], ignore_index=True)
            combined_max_fidelity_df.to_csv(max_fidelity_filename, index=False)

        if args.run_stab and result['stab'] is not None:
            sensitive_attribute = args.gender or args.age or args.race
            stab_filename = os.path.join(output_path, f'stability_{sensitive_attribute}_{bound}_{args.dataset}_{args.model_name}.csv')
            combined_stab_df = pd.concat(result['stab'], ignore_index=True)
            combined_stab_df.to_csv(stab_filename, index=False)

        if args.run_reco and result['reco'] is not None:
            sensitive_attribute = args.gender or args.age or args.race
            reco_filename = os.path.join(output_path, f'reco_{sensitive_attribute}_{bound}_{args.dataset}_{args.model_name}.csv')
            combined_reco_df = pd.concat(result['reco'], ignore_index=True)
            combined_reco_df.to_csv(reco_filename, index=False)

    # Plotting demographic parity against reductionist difference bound
    plt.figure(figsize=(10, 6))
    plt.plot(reductionist_bounds, dem_parity_values, marker='o', linestyle='--')
    plt.xlabel('Percentage violation')
    plt.ylabel('Demographic Parity')
    plt.title(f'Demographic Parity vs Percentage violation in base fairness for {args.model_name} with base {args.reductionist_type}')
    plt.grid(True)
    plot_filename = os.path.join(graph_path, f'dem_parity_plot_{args.dataset}_{args.model_name}.png')
    plt.savefig(plot_filename)

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set the global path for results
g_path = ".../results"

if rank == 0:
    # Root process to setup paths and perform initial training
    base_unfairness_results = []
    unfair_clf, bb_pred = run_training(args)
    base_unfairness = calculate_fairness_metrics(bb_pred, y_test, s_test, args.reductionist_type)
    base_unfairness_results.append(base_unfairness)

    # Save base unfairness results
    base_unfairness_df = pd.DataFrame(base_unfairness_results)
    unfairness_filename = os.path.join(g_path, f'base_unfairness_results_{args.model_name}_{args.dataset}.csv')
    base_unfairness_df.to_csv(unfairness_filename, index=False, mode='a', header=not os.path.exists(unfairness_filename))

    print(f"Base unfairness for : {args.model_name} is given by : {base_unfairness}")

    # Determine output path based on analysis type
    path_mapping = {
        'max_fidelity': {
            'COMPAS': ".../results/max_fid/compas",
            'ACSIncome': ".../results/max_fid/income",
            'ACSEmployment': ".../results/max_fid/employ",
        },
        
        'reco': {
            'COMPAS': {
                'afr': ".../results/reco/compas/g0",
                'cau': ".../results/reco/compas/g1"
            },
            'ACSIncome': {
                'male': ".../results/reco/income/g0",
                'female': ".../results/reco/income/g1"
            },
            'ACSEmployment': {
                'g0': ".../results/reco/employ/g0",
                'g1': ".../results/reco/employ/g1"
            },
        },
        'stab': {
            'COMPAS': {
                'afr': ".../results/stability/compas/g0",
                'cau': ".../results/stability/compas/g1"
            },
            'ACSIncome': {
                'male': ".../results/stability/income/g0",
                'female': ".../results/stability/income/g1"
            },
            'ACSEmployment': {
                'g0': ".../results/stability/employ/g0",
                'g1': ".../results/stability/employ/g1"
            },
        }
    }

    analysis_type = 'max_fidelity' if args.run_max_fidelity else 'reco' if args.run_reco else 'stab' if args.run_stab else None
    if analysis_type is None:
        raise ValueError("No analysis type selected.")
    
    path = path_mapping[analysis_type].get(args.dataset)
    if isinstance(path, dict):
        path = path[args.race or args.gender or args.age]

else:
    base_unfairness = None

# Broadcast base unfairness and args to all processes
base_unfairness = comm.bcast(base_unfairness, root=0)
args = comm.bcast(args, root=0)

# Split the reductionist difference bounds among processes
if rank == 0:
    jobs = split(args.reductionist_difference_bound, comm.size)
else:
    jobs = None

jobs = comm.scatter(jobs, root=0)

results = {}

for bound in jobs:
    #print(f"Processing bound: {bound}")
    dem_parity, max_fidelity_results, stab_results, reco_results = run_parallel_bound_analysis(args, bound, base_unfairness, args.seeds, g_path)
    results[bound] = {'dem_parity': dem_parity, 'max_fidelity': max_fidelity_results, 'stab': stab_results, 'reco': reco_results}

# Gather results at root
all_results = comm.gather(results, root=0)

if rank == 0:
    # Combine results from all processes
    combined_results = {}
    for result in all_results:
        combined_results.update(result)

    process_and_save_results(combined_results, args, output_path=path, graph_path=g_path)