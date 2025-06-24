import xplique
import pandas as pd
import numpy as np
import tensorflow as tf

from xplique.attributions import Lime, KernelShap

from scipy.spatial.distance import cosine, euclidean

from train_fair_blackbox import get_fair_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import copy

tf.get_logger().setLevel('ERROR')
import warnings
from scipy.stats import ConstantInputWarning
warnings.simplefilter('ignore', ConstantInputWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Metric:
    def __init__(self, model):
        self.model = model
        self.explainers = [Lime(self.model, batch_size=64, nb_samples=100), KernelShap(self.model, batch_size=64, nb_samples=100)]
        self.scaler = StandardScaler()

    def calculate_fidelity(self, black_box_preds, modified_preds, metric='acc'):
        """
        Calculates fidelity for feature attribution explanations using the specified performance metric.
        """
        if metric == 'acc':
            return accuracy_score(black_box_preds, modified_preds)
        elif metric == 'auroc':
            return roc_auc_score(black_box_preds, modified_preds, average='samples')
        elif metric == 'mse':
            return mean_squared_error(black_box_preds, modified_preds)
        else:
            raise ValueError(f"Invalid metric: {metric}. Choose from 'acc', 'auroc', or 'mse'.")

    def select(self, feature_attributions, inputs, top_features):
        """
        Selects and modifies the top k features based on feature attributions.
        """
        continuous_features = inputs.columns[inputs.nunique() > 2]
        categorical_features = inputs.columns[inputs.nunique() <= 2]

        feature_medians = inputs[continuous_features].median()
        feature_least_frequent = inputs[categorical_features].apply(lambda col: col.value_counts().idxmin())

        modified_inputs = inputs.copy()

        for i, attributions in enumerate(feature_attributions):
            num_top_features = top_features
            top_features_indices = np.argsort(attributions)[:, -num_top_features:]
            top_features_indices = np.flip(top_features_indices, axis=1)

            least_important = np.setdiff1d(np.arange(inputs.shape[1]), top_features_indices)

            for feature_idx in least_important:
                feature_name = inputs.columns[feature_idx]
                if feature_name in continuous_features:
                    variance = inputs[feature_name].var()
                    noise = np.random.normal(0, variance)
                    modified_inputs.iloc[i, feature_idx] = feature_medians[feature_name] + noise
                else:
                    modified_inputs.iloc[i, feature_idx] = feature_least_frequent[feature_name]

        return modified_inputs.values

    def max_fidelity_gap_from_average(self, samples, labels, df, sensitive_feature, k_top_features):
        """
        Calculates maximum fidelity gap from the average across subgroups for multiple metrics.
        """
        samples_df = pd.DataFrame(samples, columns=df.columns)
        samples_df[sensitive_feature] = samples_df[sensitive_feature].astype(int)
        unique_groups = samples_df[sensitive_feature].unique()
        scaled_samples = self.scaler.fit_transform(samples)
        labels = tf.cast(labels, tf.float64)[:, np.newaxis]

        metrics = ['acc']
        results = []
        explanations = []

        for explainer in self.explainers:
            explainer_name = explainer.__class__.__name__

            explanation = explainer(scaled_samples, labels)
            explanations.append(explanation)

            expl_inputs = self.select(feature_attributions=explanations, inputs=samples_df, top_features=k_top_features)
            expl_preds = self.model.predict(expl_inputs)[:, np.newaxis]
            black_box_preds = self.model.predict(scaled_samples)[:, np.newaxis]

            for metric in metrics:
                overall_fidelity = self.calculate_fidelity(black_box_preds, expl_preds, metric=metric)
                group_fidelities = {}
                group_gaps = {}

                for group in unique_groups:
                    group_indices = samples_df[samples_df[sensitive_feature] == group].index
                    valid_group_indices = group_indices[group_indices < samples.shape[0]]

                    group_black_box_preds = black_box_preds[valid_group_indices]
                    group_expl_preds = expl_preds[valid_group_indices]

                    group_fidelity = self.calculate_fidelity(group_black_box_preds, group_expl_preds, metric=metric)
                    group_fidelities[group] = group_fidelity
                    fidelity_gap = abs(overall_fidelity - group_fidelity)
                    group_gaps[group] = fidelity_gap

                max_fidelity_gap = max(group_gaps.values())

                result = {
                    'Method name': explainer_name,
                    'Metric': metric,
                    'Max Error gap from average': max_fidelity_gap
                }
                for group in unique_groups:
                    result[f'Group {group} Fidelity Gaps'] = group_gaps[group]

                results.append(result)

        return pd.DataFrame(results)

    '''def mean_fidelity_gap_amongst_subgroups(self, samples, labels, df, sensitive_feature, k_top_features):
        """
        Calculates mean fidelity gap amongst subgroups for multiple metrics.
        """
        samples_df = pd.DataFrame(samples, columns=df.columns)
        samples_df[sensitive_feature] = samples_df[sensitive_feature].astype(int)
        unique_groups = samples_df[sensitive_feature].unique()
        G = len(unique_groups)

        scaled_samples = self.scaler.fit_transform(samples)
        labels = tf.cast(labels, tf.float64)[:, np.newaxis]

        metrics = ['acc', 'auroc', 'mse']
        results = []
        explanations = []

        for explainer in self.explainers:
            explainer_name = explainer.__class__.__name__

            explanation = explainer(scaled_samples, labels)
            explanations.append(explanation)

            expl_inputs = self.select(feature_attributions=explanations, inputs=samples_df, top_features=k_top_features)
            expl_preds = self.model.predict(expl_inputs)[:, np.newaxis]
            black_box_preds = self.model.predict(scaled_samples)[:, np.newaxis]

            for metric in metrics:
                subgroup_fidelities = []

                for k in range(G):
                    for j in range(k + 1, G):
                        group_k = unique_groups[k]
                        group_j = unique_groups[j]

                        group_k_indices_np = df[df[sensitive_feature] == group_k].index.to_numpy()
                        group_j_indices_np = df[df[sensitive_feature] == group_j].index.to_numpy()

                        group_k_indices_np = group_k_indices_np[group_k_indices_np < samples.shape[0]]
                        group_j_indices_np = group_j_indices_np[group_j_indices_np < samples.shape[0]]

                        if len(group_k_indices_np) == 0 or len(group_j_indices_np) == 0:
                            continue

                        group_k_black_box_preds = black_box_preds[group_k_indices_np]
                        group_j_black_box_preds = black_box_preds[group_j_indices_np]
                        group_k_explanation_preds = expl_preds[group_k_indices_np]
                        group_j_explanation_preds = expl_preds[group_j_indices_np]

                        group_k_fidelity = self.calculate_fidelity(group_k_black_box_preds, group_k_explanation_preds, metric=metric)
                        group_j_fidelity = self.calculate_fidelity(group_j_black_box_preds, group_j_explanation_preds, metric=metric)

                        subgroup_fidelities.append(abs(group_k_fidelity - group_j_fidelity))

                mean_fidelity_gap = (2 / (G * (G - 1))) * sum(subgroup_fidelities)

                results.append({
                    'Method name': explainer_name,
                    'Metric': metric,
                    'Mean Fidelity Gap Amongst Subgroups': mean_fidelity_gap
                })

        return pd.DataFrame(results)'''


    def reco(self, gender_samples, gender_labels, n_explanations=5, distance_metric='euclidean'):
        """
        Calculate the inconsistency of explanations for each method.

        Parameters:
        gender_samples (tf.Tensor): Input samples.
        gender_labels (tf.Tensor): Labels for the input samples.
        n_explanations (int): Number of explanations to generate for inconsistency measurement.
        distance_metric (str): Distance metric to use ('euclidean' or 'cosine').

        Returns:
        pd.DataFrame: Inconsistency scores for each method.
        """
        inconsistency_scores = []
    
        gender_samples = self.scaler.fit_transform(gender_samples)

        # Cast to the required type
        gender_samples, gender_labels = tf.cast(gender_samples, tf.float64), tf.cast(gender_labels, tf.float64)
        gender_labels = tf.expand_dims(gender_labels, axis=-1)

        for explainer in self.explainers:
            explainer_name = explainer.__class__.__name__

            explanations_list = []
            for _ in range(n_explanations):
                explanations = explainer(gender_samples, gender_labels)
                explanations_list.append(explanations)

            # Calculate pairwise distances between explanations
            pairwise_distances = []
            for i in range(n_explanations):
                for j in range(i + 1, n_explanations):
                    if distance_metric == 'euclidean':
                        distance = np.linalg.norm(explanations_list[i] - explanations_list[j])
                    elif distance_metric == 'cosine':
                        distance = cosine(explanations_list[i].flatten(), explanations_list[j].flatten())
                    else:
                        raise ValueError("Unsupported distance metric. Use 'euclidean' or 'cosine'.")
                    pairwise_distances.append(distance)

            # Calculate the Inconsistency Score using the provided formula
            inconsistency_score = (2 / (n_explanations * (n_explanations - 1))) * sum(pairwise_distances)
            inconsistency_scores.append((explainer_name, inconsistency_score))

        return pd.DataFrame(inconsistency_scores, columns=["Method name", "Inconsistency score"])

    def generate_noisy_points(self, x, m):
        """
        Generate m noisy versions of the input point x.
        
        Parameters:
        - x: Original data point (numpy array)
        - m: Number of noisy points to generate
        
        Returns:
        - noisy_points: Array of noisy points
        """
        variances = np.var(x, axis=0)
        noisy_points = np.array([x + np.random.normal(0, variances, size=x.shape) for _ in range(m)])
        return noisy_points
    
    def calculate_explanation(self, x, y, explainer):
        """
        Calculate the explanation for a given point x using the provided explainer.
        
        Parameters:
        - x: Data point (numpy array)
        - explainer: The explainer object used for generating explanations
        
        Returns:
        - explanation: Explanation for the input point x
        """
        x_scaled = self.scaler.fit_transform(x)
        x_cast, y_cast = tf.cast(x_scaled, tf.float64), tf.cast(y, tf.float64)
        y_cast = tf.expand_dims(y_cast, axis=-1)
        
        explanation = explainer(x_cast, y_cast)
        return explanation
    
    def stab(self, x, y, m = 4):
        """
        Calculate the instability of the explanation for a given point x.
        
        Parameters:
        - x: Original data point (numpy array)
        - m: Number of noisy points to generate
        - n_explanations: Number of explanations to generate
        
        Returns:
        - instabilities: Dictionary of instability values for each explainer
        """
        instabilities = {}
        noisy_points = self.generate_noisy_points(x, m)

        for explainer in self.explainers:
            explainer_name = explainer.__class__.__name__
            total_l1_distance = 0.0
            
            original_explanation = self.calculate_explanation(x, y, explainer)
            
            for noisy_point in noisy_points:
                noisy_explanation = self.calculate_explanation(noisy_point, y, explainer)
                l1_distance = np.sum(np.abs(original_explanation - noisy_explanation))
                total_l1_distance += l1_distance
            
            instability = total_l1_distance / m
            instabilities[explainer_name] = instability
            
        instabilities_df = pd.DataFrame(list(instabilities.items()), columns=['Method name', 'Instability'])
        return instabilities_df
