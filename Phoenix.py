"""
Phoenix functional GitHub version aligned with the paper-level method description.

This version focuses on functional consistency with the current Phoenix-OI manuscript:
- default single-objective scalarized wrapper fitness;
- 5-fold KNN evaluator with k=3 by default;
- fixed lambda penalty = 0.001 by default;
- strict No-ANOVA default, with ANOVA prefilter only as an optional switch;
- deterministic binarization using threshold > 0.5;
- OED/MED initialization, IMM-style movement inference, and elite-only two-stage refinement;
- stored-fitness elite refinement in the primary scalarized mode, without redundant full-population reevaluation;
- exact stage-wise FFE accounting, where one nonempty wrapper-fitness call (one complete CV evaluation) equals one FFE;
- optional GPSISF-style scoring and post-hoc Pareto analysis.

Not included in this lightweight GitHub script:
- ablation variant runner;
- multi-run statistical analysis pipeline;
- full non-dominated archive implementation for the optional multi-objective mode.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from oa_package.oa import OrthogonalArray


class GPPSO:
    def __init__(
        self,
        feature,
        feature_names,
        label,
        w=0.9,
        c1=2.0,
        c2=2.0,
        Np=30,
        search_space=200,
        max_iter=30,
        max_conv_iter=10,
        tol=0.001,
        rho=3,
        fitness_type="original",
        lambda_penalty=0.001,
        knn_k=3,
        cv=5,
        use_anova_prefilter=False,
        remove_constant_features=True,
        save_pareto=False,
        output_dir="results",
        dataset_name="unknown_dataset",
        random_state=None,
    ):
        assert len(feature) == len(label), "feature and label must have the same number of samples."
        if fitness_type not in {"original", "gpsisf"}:
            raise ValueError("fitness_type must be either 'original' or 'gpsisf'.")

        self.feature = np.asarray(feature)
        self.feature_names = np.asarray(feature_names)
        self.label = np.asarray(label)

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.Np = int(Np)
        self.D = self.feature.shape[1]
        self.search_space = int(search_space)
        self.max_iter = int(max_iter)
        self.max_conv_iter = int(max_conv_iter)
        self.tol = tol
        self.rho = int(rho)
        self.fitness_type = fitness_type
        self.lambda_penalty = float(lambda_penalty)
        self.knn_k = int(knn_k)
        self.cv = int(cv)
        self.use_anova_prefilter = bool(use_anova_prefilter)
        self.remove_constant_features = bool(remove_constant_features)

        self.save_pareto = bool(save_pareto)
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # For GPSISF scoring and post-hoc Pareto analysis.
        self.population_objectives = []
        self.pareto_solutions = []
        self.pareto_metrics = []
        self.iteration_history = []

        # Exact search-FFE accounting used in the manuscript.
        # One FFE = one call to the exact wrapper fitness function for a
        # nonempty candidate subset. With the default cv=5, one FFE therefore
        # contains one complete 5-fold CV evaluation internally; the five folds
        # are NOT counted as five separate FFEs.
        self.ffe_stage_names = (
            "initialization_oa",
            "population_initial",
            "imm_oa",
            "population_after_movement",
            "elite_recombination_oa",
            "elite_recombination_candidate",
            "elite_reduction_oa",
            "elite_reduction_candidate",
        )
        self.ffe_counts = {stage: 0 for stage in self.ffe_stage_names}

        if self.save_pareto:
            self.dataset_output_dir = os.path.join(self.output_dir, self.dataset_name)
            os.makedirs(self.dataset_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Evaluation and preprocessing
    # ------------------------------------------------------------------
    def reset_ffe_counts(self):
        """Reset all manuscript-defined search-stage FFE counters."""
        self.ffe_counts = {stage: 0 for stage in self.ffe_stage_names}

    def _record_ffe(self, stage):
        """
        Record one exact wrapper-fitness call.

        stage=None is reserved for reporting/post-hoc analysis and is excluded
        from search-FFE accounting.
        """
        if stage is None:
            return
        if stage not in self.ffe_counts:
            raise ValueError(
                f"Unknown FFE stage '{stage}'. Expected one of: "
                + ", ".join(self.ffe_stage_names)
            )
        self.ffe_counts[stage] += 1

    def total_search_ffe(self):
        """Return the total number of search FFEs across all eight stages."""
        return int(sum(self.ffe_counts.values()))

    def get_ffe_report(self):
        """Return stage-wise and Table-III-style grouped FFE counts."""
        report = {stage: int(self.ffe_counts[stage]) for stage in self.ffe_stage_names}
        report["total_search_ffe"] = self.total_search_ffe()
        report["cv_folds_per_ffe"] = int(self.cv)
        report["internal_cv_fold_evaluations"] = int(self.total_search_ffe() * self.cv)

        # Grouped categories used in the manuscript stage-wise FFE table.
        report["group_initialization_oi"] = report["initialization_oa"]
        report["group_imm"] = report["imm_oa"]
        report["group_elite_refinement"] = (
            report["elite_recombination_oa"]
            + report["elite_recombination_candidate"]
            + report["elite_reduction_oa"]
            + report["elite_reduction_candidate"]
        )
        report["group_other"] = (
            report["population_initial"]
            + report["population_after_movement"]
        )
        return report

    def print_ffe_report(self):
        """Print stage-wise FFE counts and their grouped manuscript totals."""
        report = self.get_ffe_report()
        print("\n" + "=" * 68)
        print("Exact search-FFE report")
        print("=" * 68)
        print(
            f"Definition: 1 FFE = 1 wrapper fitness-function call = "
            f"one complete {self.cv}-fold CV evaluation"
        )
        for stage in self.ffe_stage_names:
            print(f"{stage:36s}: {report[stage]}")
        print("-" * 68)
        print(f"{'Total search FFEs':36s}: {report['total_search_ffe']}")
        print(
            f"{'Internal CV fold evaluations':36s}: "
            f"{report['internal_cv_fold_evaluations']}"
        )
        print("\nGrouped counts (Table-III style):")
        print(f"{'Initialization OI':36s}: {report['group_initialization_oi']}")
        print(f"{'IMM':36s}: {report['group_imm']}")
        print(f"{'Elite refinement':36s}: {report['group_elite_refinement']}")
        print(f"{'Other':36s}: {report['group_other']}")
        print("=" * 68)

    def preprocess_features(self):
        """
        Strict paper-consistent default:
        - remove constant features only;
        - do NOT use ANOVA SelectKBest unless use_anova_prefilter=True.
        """
        data = self.feature
        feature_names = self.feature_names

        if self.remove_constant_features:
            selector = VarianceThreshold()
            data = selector.fit_transform(data)
            mask = selector.get_support()
            feature_names = feature_names[mask]

        if self.use_anova_prefilter and data.shape[1] > self.search_space:
            k = min(self.search_space, data.shape[1])
            selector = SelectKBest(f_classif, k=k)
            data = selector.fit_transform(data, self.label)
            selected_feature_indices = selector.get_support(indices=True)
            feature_names = feature_names[selected_feature_indices]

        self.feature = data
        self.feature_names = feature_names
        self.D = self.feature.shape[1]

        if self.D == 0:
            raise ValueError("No valid feature remains after preprocessing.")

    def binarize(self, x):
        """Deterministic paper-consistent binarization: selected iff x_j > 0.5."""
        return (np.clip(np.asarray(x), 0, 1) > 0.5).astype(int)

    def selected_indices(self, selected_features):
        return np.where(np.asarray(selected_features) > 0.5)[0]

    def cv_accuracy(self, selected_features):
        selected_indices = self.selected_indices(selected_features)
        if len(selected_indices) == 0:
            return -np.inf

        X_prime = self.feature[:, selected_indices]
        classifier = KNeighborsClassifier(n_neighbors=self.knn_k)
        scores = cross_val_score(classifier, X_prime, self.label, cv=self.cv, scoring="accuracy")
        return float(np.mean(scores))

    def calculate_objectives(self, selected_features):
        """Return (accuracy, number_of_selected_features)."""
        selected_indices = self.selected_indices(selected_features)
        if len(selected_indices) == 0:
            return (0.0, 0)
        accuracy = self.cv_accuracy(selected_features)
        return (accuracy, len(selected_indices))

    def fitness_original(self, selected_features):
        """
        Primary paper mode:
            Fitness(S) = Acc_CV(S) - lambda * |S|
        """
        selected_indices = self.selected_indices(selected_features)
        if len(selected_indices) == 0:
            return -np.inf
        return self.cv_accuracy(selected_features) - self.lambda_penalty * len(selected_indices)

    def dominates_paper_definition(self, obj1, obj2):
        """
        obj = (accuracy, num_features)
        Maximize accuracy and minimize number of selected features.
        """
        accuracy1, num_features1 = obj1
        accuracy2, num_features2 = obj2

        all_better_or_equal = (accuracy1 >= accuracy2) and (num_features1 <= num_features2)
        at_least_one_strictly_better = (accuracy1 > accuracy2) or (num_features1 < num_features2)
        return all_better_or_equal and at_least_one_strictly_better

    def gpsisf_score_from_objectives(self, current_objectives):
        """Score one already-evaluated objective pair against the current population."""
        if not self.population_objectives:
            return current_objectives[0]

        p = 0
        q = 0
        for other_objectives in self.population_objectives:
            if self.dominates_paper_definition(current_objectives, other_objectives):
                p += 1
            elif self.dominates_paper_definition(other_objectives, current_objectives):
                q += 1

        c = len(self.population_objectives)
        return p - q + c

    def fitness_gpsisf_original(self, selected_features):
        """
        Optional GPSISF-style scoring:
            GPSISF(X) = p - q + c
        This script uses GPSISF as scalar ranking support, not as a full archive-based MOEA.
        Each candidate objective pair is evaluated once before dominance scoring.
        """
        if len(self.selected_indices(selected_features)) == 0:
            return -np.inf

        current_objectives = self.calculate_objectives(selected_features)
        return self.gpsisf_score_from_objectives(current_objectives)

    def update_population_objectives(self, x, stage=None):
        if self.fitness_type != "gpsisf":
            return

        objectives = []
        for i in range(self.Np):
            candidate = x[:, i]
            if len(self.selected_indices(candidate)) == 0:
                objectives.append((0.0, 0))
                continue
            self._record_ffe(stage)
            objectives.append(self.calculate_objectives(candidate))
        self.population_objectives = objectives

    def evaluate_population_fitness(self, x, stage):
        """
        Evaluate a complete population once per member.

        Every nonempty member causes one exact wrapper-fitness call and
        increments the supplied search-stage counter by one.
        """
        if self.fitness_type == "gpsisf":
            self.update_population_objectives(x, stage=stage)
            return np.asarray(
                [self.gpsisf_score_from_objectives(obj) for obj in self.population_objectives],
                dtype=float,
            )

        return np.asarray(
            [self.fitness(x[:, i], stage=stage) for i in range(self.Np)],
            dtype=float,
        )

    def fitness(self, selected_features, stage=None):
        """
        Evaluate one candidate with the exact wrapper fitness.

        Search accounting rule:
            one nonempty wrapper-fitness call = one FFE, regardless of the
            number of CV folds executed internally.
        """
        if len(self.selected_indices(selected_features)) == 0:
            return -np.inf

        self._record_ffe(stage)

        if self.fitness_type == "gpsisf":
            return self.fitness_gpsisf_original(selected_features)
        return self.fitness_original(selected_features)

    # ------------------------------------------------------------------
    # Pareto analysis helpers: optional post-hoc analysis only
    # ------------------------------------------------------------------
    def update_pareto_solutions(self, x, iter_num):
        if not self.save_pareto:
            return

        current_solutions = []
        current_metrics = []
        for i in range(self.Np):
            solution = self.binarize(x[:, i])
            if np.sum(solution) > 0:
                current_solutions.append(solution.copy())
                current_metrics.append(self.calculate_objectives(solution))

        self.pareto_solutions.extend(current_solutions)
        self.pareto_metrics.extend(current_metrics)

        if current_metrics:
            best_idx = int(np.argmax([m[0] for m in current_metrics]))
            self.iteration_history.append(
                {
                    "iteration": iter_num,
                    "best_solution": current_solutions[best_idx],
                    "metrics": current_metrics[best_idx],
                }
            )

    def find_pareto_front(self, solutions, metrics):
        pareto_indices = []
        for i, metric1 in enumerate(metrics):
            is_pareto = True
            for j, metric2 in enumerate(metrics):
                if i != j and self.dominates_paper_definition(metric2, metric1):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_indices.append(i)
        return pareto_indices

    def save_pareto_data(self):
        if not self.pareto_metrics or not self.save_pareto:
            return

        pareto_indices = set(self.find_pareto_front(self.pareto_solutions, self.pareto_metrics))
        rows = []
        for i, (solution, metrics) in enumerate(zip(self.pareto_solutions, self.pareto_metrics)):
            selected = self.selected_indices(solution)
            selected_feature_names = self.feature_names[selected] if len(selected) > 0 else []
            rows.append(
                {
                    "Solution_ID": i,
                    "Is_Pareto": i in pareto_indices,
                    "Accuracy": metrics[0],
                    "Num_Features": int(metrics[1]),
                    "Selected_Features": ",".join(map(str, selected_feature_names)),
                    "Feature_Indices": ",".join(map(str, selected)),
                }
            )

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.dataset_output_dir, "pareto_solutions.csv")
        df.to_csv(csv_path, index=False)

        pareto_csv_path = os.path.join(self.dataset_output_dir, "pareto_front_only.csv")
        df[df["Is_Pareto"]].to_csv(pareto_csv_path, index=False)
        print(f"Pareto solutions saved to: {csv_path}")
        print(f"Pareto front saved to: {pareto_csv_path}")

    def plot_pareto_front_2d(self):
        if not self.pareto_metrics:
            return

        accuracies = [m[0] for m in self.pareto_metrics]
        num_features = [m[1] for m in self.pareto_metrics]
        pareto_indices = self.find_pareto_front(self.pareto_solutions, self.pareto_metrics)

        plt.figure(figsize=(10, 8))
        plt.scatter(num_features, accuracies, alpha=0.6, label="All Solutions", s=30)

        if pareto_indices:
            pareto_features = [num_features[i] for i in pareto_indices]
            pareto_accuracies = [accuracies[i] for i in pareto_indices]
            plt.scatter(pareto_features, pareto_accuracies, label="Pareto Front", s=60, marker="^")
            if len(pareto_indices) > 1:
                sorted_pairs = sorted(zip(pareto_features, pareto_accuracies))
                sorted_features, sorted_accuracies = zip(*sorted_pairs)
                plt.plot(sorted_features, sorted_accuracies, "--", alpha=0.7)

        plt.xlabel("Number of Features")
        plt.ylabel("Accuracy")
        plt.title(f"Pareto Front: Accuracy vs Number of Features ({self.dataset_name})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if self.save_pareto:
            fig_path = os.path.join(self.dataset_output_dir, "pareto_front_2d.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Pareto plot saved to: {fig_path}")
        plt.show()

    def generate_pareto_report(self):
        if not self.pareto_metrics:
            return
        pareto_indices = self.find_pareto_front(self.pareto_solutions, self.pareto_metrics)
        print("\n" + "=" * 60)
        print(f"Pareto front report - {self.dataset_name}")
        print("=" * 60)
        print(f"Total collected solutions: {len(self.pareto_solutions)}")
        print(f"Pareto front solutions: {len(pareto_indices)}")
        if pareto_indices:
            pareto_metrics = [self.pareto_metrics[i] for i in pareto_indices]
            accuracies = [m[0] for m in pareto_metrics]
            features = [m[1] for m in pareto_metrics]
            print(f"Accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f}")
            print(f"Feature-count range: {int(min(features))} - {int(max(features))}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # OED/MED core
    # ------------------------------------------------------------------
    def random_cut_points(self, vector_length, num_segments):
        """Return unique integer cut points that form nonempty contiguous segments."""
        vector_length = int(vector_length)
        num_segments = max(1, min(int(num_segments), vector_length))
        if num_segments <= 1:
            return []
        boundaries = np.arange(1, vector_length)
        cuts = np.random.choice(boundaries, size=num_segments - 1, replace=False)
        return sorted(int(c) for c in cuts)

    def split_x(self, x, cut_points):
        x = list(x)
        segments = []
        prev_point = 0
        for point_index in cut_points:
            point_index = int(point_index)
            segments.append(x[prev_point:point_index])
            prev_point = point_index
        segments.append(x[prev_point:])
        return segments

    def _mean_or_neg_inf(self, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return -np.inf
        return float(np.mean(values))

    def _assemble_from_row(self, orthogonal_table, row_idx, x_g_segments, x_l_segments, num_segments):
        combination = []
        for j in range(num_segments):
            if orthogonal_table.data[row_idx][j] == 1:
                combination.extend(x_g_segments[j])
            else:
                combination.extend(x_l_segments[j])
        return self.binarize(combination)

    def _repair_empty_solution(self, candidate, x_g_segments, x_l_segments, main_effects):
        """
        If MED inference returns an empty solution, activate the non-empty segment-level
        choice with strongest main-effect support. Ties prefer smaller segment size.
        """
        candidate = self.binarize(candidate)
        if np.sum(candidate) > 0:
            return candidate

        choices = []
        offset = 0
        for seg_idx, (seg_g, seg_l) in enumerate(zip(x_g_segments, x_l_segments)):
            for level, segment in [(1, seg_g), (0, seg_l)]:
                segment = np.asarray(segment, dtype=int)
                if np.sum(segment) > 0:
                    me = main_effects.get((seg_idx, level), -np.inf)
                    choices.append((me, -int(np.sum(segment)), offset, segment))
            offset += len(seg_g)

        if not choices:
            return candidate

        _, _, offset, segment = max(choices, key=lambda item: (item[0], item[1]))
        repaired = np.zeros_like(candidate)
        repaired[offset : offset + len(segment)] = segment
        return self.binarize(repaired)

    def calculate_MED(
        self,
        orthogonal_table,
        x_g_segments,
        x_l_segments,
        num_segments,
        ffe_stage,
    ):
        """
        MED segment inference between two proposal sources.
        Level 1 = x_g segment; level 0 = x_l segment.
        Tie rule: prefer the smaller segment; if still tied, prefer x_g as the elite/global side.
        """
        trial_fitness = np.full(len(orthogonal_table.data), -np.inf, dtype=float)
        trial_solutions = []

        for row_idx in range(len(orthogonal_table.data)):
            combination = self._assemble_from_row(
                orthogonal_table, row_idx, x_g_segments, x_l_segments, num_segments
            )
            trial_solutions.append(combination)
            if np.sum(combination) > 0:
                trial_fitness[row_idx] = self.fitness(combination, stage=ffe_stage)

        best_row = int(np.nanargmax(trial_fitness))
        best_oa_row = trial_solutions[best_row]

        best_combination = []
        main_effects = {}
        for seg_idx in range(num_segments):
            rows_level_1 = [
                r for r in range(len(orthogonal_table.data)) if orthogonal_table.data[r][seg_idx] == 1
            ]
            rows_level_0 = [
                r for r in range(len(orthogonal_table.data)) if orthogonal_table.data[r][seg_idx] != 1
            ]
            me1 = self._mean_or_neg_inf(trial_fitness[rows_level_1])
            me0 = self._mean_or_neg_inf(trial_fitness[rows_level_0])
            main_effects[(seg_idx, 1)] = me1
            main_effects[(seg_idx, 0)] = me0

            seg_g = np.asarray(x_g_segments[seg_idx], dtype=int)
            seg_l = np.asarray(x_l_segments[seg_idx], dtype=int)

            if me1 > me0:
                chosen = seg_g
            elif me0 > me1:
                chosen = seg_l
            else:
                # Paper-aligned tie handling: prefer smaller subset; if still tied, prefer x_g.
                if np.sum(seg_g) <= np.sum(seg_l):
                    chosen = seg_g
                else:
                    chosen = seg_l
            best_combination.extend(chosen)

        best_combination = self._repair_empty_solution(
            best_combination, x_g_segments, x_l_segments, main_effects
        )
        return best_oa_row, best_combination

    def init_split_x(self, x):
        segments = []
        zero_segments = []
        segment = []
        zero_segment = []
        for val in x:
            segment.append(int(val))
            zero_segment.append(0)
            if val == 1:
                segments.append(segment)
                zero_segments.append(zero_segment)
                segment = []
                zero_segment = []
        if segment:
            segments.append(segment)
            zero_segments.append(zero_segment)
        return segments, zero_segments

    def init_calculate_MED(self, orthogonal_table, x_g_segments, x_l_segments, init_feature, features_index):
        trial_fitness = np.full(len(orthogonal_table.data), -np.inf, dtype=float)
        trial_solutions = []

        for row_idx in range(len(orthogonal_table.data)):
            combination = self._assemble_from_row(
                orthogonal_table, row_idx, x_g_segments, x_l_segments, init_feature
            )
            trial_solutions.append(combination)
            if np.sum(combination) > 0:
                trial_fitness[row_idx] = self.fitness(combination, stage="initialization_oa")

        sorted_rows = np.argsort(trial_fitness)[::-1]
        best_combinations = []
        for row_idx in sorted_rows[: max(0, self.Np - 1)]:
            if np.isfinite(trial_fitness[row_idx]) and np.sum(trial_solutions[row_idx]) > 0:
                best_combinations.append(trial_solutions[row_idx].copy())

        while len(best_combinations) < self.Np - 1:
            r = np.zeros(self.D, dtype=int)
            for feature in features_index:
                r[feature] = np.random.choice([0, 1])
            if np.sum(r) == 0:
                r[np.random.choice(features_index)] = 1
            best_combinations.append(r)

        # Main-effect inferred initializer, used as the final population member.
        best = []
        main_effects = {}
        for seg_idx in range(init_feature):
            rows_level_1 = [
                r for r in range(len(orthogonal_table.data)) if orthogonal_table.data[r][seg_idx] == 1
            ]
            rows_level_0 = [
                r for r in range(len(orthogonal_table.data)) if orthogonal_table.data[r][seg_idx] != 1
            ]
            me1 = self._mean_or_neg_inf(trial_fitness[rows_level_1])
            me0 = self._mean_or_neg_inf(trial_fitness[rows_level_0])
            main_effects[(seg_idx, 1)] = me1
            main_effects[(seg_idx, 0)] = me0

            seg_g = np.asarray(x_g_segments[seg_idx], dtype=int)
            seg_l = np.asarray(x_l_segments[seg_idx], dtype=int)
            if me1 > me0:
                chosen = seg_g
            elif me0 > me1:
                chosen = seg_l
            else:
                # Tie: prefer fewer selected features; if still tied, prefer retaining the feature.
                chosen = seg_l if np.sum(seg_l) < np.sum(seg_g) else seg_g
            best.extend(chosen)

        best = self._repair_empty_solution(best, x_g_segments, x_l_segments, main_effects)
        best_combinations.append(best)
        return best_combinations

    def loss_feature_calculate_MED(self, orthogonal_table, x_g_segments, x_l_segments, init_feature):
        """
        Stage-2 MED retain/remove reduction.
        Level 1 = retain current segment; level 0 = replace by zero segment.
        Tie rule follows the paper equation: retain if ME_1 >= ME_0.
        """
        trial_fitness = np.full(len(orthogonal_table.data), -np.inf, dtype=float)
        for row_idx in range(len(orthogonal_table.data)):
            combination = self._assemble_from_row(
                orthogonal_table, row_idx, x_g_segments, x_l_segments, init_feature
            )
            if np.sum(combination) > 0:
                trial_fitness[row_idx] = self.fitness(combination, stage="elite_reduction_oa")

        reduced = []
        for seg_idx in range(init_feature):
            rows_level_1 = [
                r for r in range(len(orthogonal_table.data)) if orthogonal_table.data[r][seg_idx] == 1
            ]
            rows_level_0 = [
                r for r in range(len(orthogonal_table.data)) if orthogonal_table.data[r][seg_idx] != 1
            ]
            me1 = self._mean_or_neg_inf(trial_fitness[rows_level_1])
            me0 = self._mean_or_neg_inf(trial_fitness[rows_level_0])

            if me1 >= me0:
                reduced.extend(x_g_segments[seg_idx])
            else:
                reduced.extend(np.zeros_like(x_l_segments[seg_idx], dtype=int))

        return self.binarize(reduced)

    def initialization(self, features_index):
        x_all = np.zeros(self.D, dtype=int)
        for feature in features_index:
            x_all[feature] = 1

        x_1_segments, x_0_segments = self.init_split_x(x_all)
        init_feature = len(x_1_segments)
        init_orthogonal_table = OrthogonalArray(init_feature)
        population = self.init_calculate_MED(
            init_orthogonal_table, x_1_segments, x_0_segments, init_feature, features_index
        )
        return np.array(population, dtype=int).T

    # ------------------------------------------------------------------
    # Phoenix evolution and elite refinement
    # ------------------------------------------------------------------
    def local_search(self, x, orthogonal_table, features_index, ranking, num_segments, fit):
        """
        Perform one elite-only two-stage refinement pass.

        The incumbent exact fitness is reused from ``fit``. Each inferred candidate is
        evaluated once, and an accepted solution and its exact fitness are written back
        together before reranking. In the primary scalarized mode this avoids any
        complete-population reevaluation after refinement.
        """
        improved = False
        elite_idx = int(ranking[0])
        elite = self.binarize(x[:, elite_idx])
        elite_fit = float(fit[elite_idx])

        # Stage 1: OED-guided local recombination with complementary parent.
        complement = elite.copy()
        for feature in features_index:
            complement[feature] = 1 - complement[feature]

        cut_points = self.random_cut_points(len(elite), num_segments)
        elite_segments = self.split_x(elite, cut_points)
        complement_segments = self.split_x(complement, cut_points)

        _, ls_best_combination = self.calculate_MED(
            orthogonal_table,
            elite_segments,
            complement_segments,
            num_segments,
            ffe_stage="elite_recombination_oa",
        )
        candidate_fit = self.fitness(
            ls_best_combination, stage="elite_recombination_candidate"
        )
        if candidate_fit > elite_fit:
            elite = self.binarize(ls_best_combination)
            x[:, elite_idx] = elite
            elite_fit = float(candidate_fit)
            improved = True

        # Stage 2: MED-guided retain/remove compactness reduction.
        if np.sum(elite) > 2:
            retain_segments, zero_segments = self.init_split_x(elite)
            reduction_orthogonal_table = OrthogonalArray(len(retain_segments))
            reduced = self.loss_feature_calculate_MED(
                reduction_orthogonal_table, retain_segments, zero_segments, len(retain_segments)
            )
            if np.sum(reduced) > 0:
                reduced_fit = self.fitness(
                    reduced, stage="elite_reduction_candidate"
                )
                if reduced_fit > elite_fit:
                    elite = self.binarize(reduced)
                    x[:, elite_idx] = elite
                    elite_fit = float(reduced_fit)
                    improved = True

        fit[elite_idx] = elite_fit
        ranking = np.argsort(fit)[::-1]
        return x, fit, ranking, improved

    def Evolve(self, x, r1, r2, v, orthogonal_table, imm_count, ranking, num_segments, pbest, gbest):
        # IMM-updated particles.
        for particle in range(0, imm_count):
            idx = ranking[particle]
            v_global = np.multiply((gbest - x[:, idx]), r2[:, idx]) * self.c2
            v_local = np.multiply((pbest[:, idx] - x[:, idx]), r1[:, idx]) * self.c1
            v[:, idx] = x[:, idx] - v[:, idx]

            x_g = self.binarize(x[:, idx] + self.w * v[:, idx] + v_global)
            x_l = self.binarize(x[:, idx] + self.w * v[:, idx] + v_local)

            cut_points = self.random_cut_points(len(x_g), num_segments)
            x_g_segments = self.split_x(x_g, cut_points)
            x_l_segments = self.split_x(x_l, cut_points)

            _, best_combination = self.calculate_MED(
                orthogonal_table,
                x_g_segments,
                x_l_segments,
                num_segments,
                ffe_stage="imm_oa",
            )
            x[:, idx] = self.binarize(best_combination)

        # Diversification / standard update particles.
        for particle in range(imm_count, self.Np):
            idx = ranking[particle]
            if np.random.rand() < 0.5:
                proposal = (
                    x[:, idx]
                    + self.w * v[:, idx]
                    + np.multiply((gbest - x[:, idx]), r2[:, idx]) * self.c2
                    + np.multiply((pbest[:, idx] - x[:, idx]), r1[:, idx]) * self.c1
                )
                x[:, idx] = self.binarize(proposal)
            else:
                x[:, idx] = np.random.choice([0, 1], size=self.D, p=[0.7, 0.3])
                if np.sum(x[:, idx]) == 0:
                    x[np.random.randint(0, self.D), idx] = 1

        return x, v

    def update_p(self, fit, pbest_val, pbest, x):
        update_indices = fit > pbest_val
        pbest_val[update_indices] = np.copy(fit[update_indices])
        pbest[:, update_indices] = np.copy(x[:, update_indices])
        return pbest_val, pbest

    def update_g(self, best_selected_features, best_gbest_val, gbest_val, gbest_store, pbest_val, iteration, pbest, conv_gen):
        gbest_idx = int(np.argmax(pbest_val))
        gbest_val[iteration] = pbest_val[gbest_idx]
        gbest = np.copy(pbest[:, gbest_idx])
        gbest_store[:, iteration] = gbest

        if gbest_val[iteration] - best_gbest_val < self.tol:
            conv_gen += 1
        else:
            conv_gen = 0

        if gbest_val[iteration] > best_gbest_val:
            best_gbest_val = gbest_val[iteration]
            best_selected_features = gbest.copy()

        return conv_gen, best_gbest_val, best_selected_features, gbest_val, gbest_store, gbest

    def show_iter(self, best_selected_features, best_gbest_val, conv_gen, iteration):
        selected_count = len(self.selected_indices(best_selected_features))
        if self.fitness_type == "original" and np.isfinite(best_gbest_val):
            # Recover accuracy from the exact scalarized fitness without another CV call.
            acc = best_gbest_val + self.lambda_penalty * selected_count
        else:
            acc, _ = self.calculate_objectives(best_selected_features)
        print(
            f"Iter.={iteration}, gbest_val={best_gbest_val:.6f}, "
            f"acc={acc:.4f}, selected_features={selected_count}, conv_iter={conv_gen}"
        )

    def show(self, best_selected_features, best_gbest_val):
        best_selected_features = self.binarize(best_selected_features)
        selected = self.selected_indices(best_selected_features)
        selected_feature_names = self.feature_names[selected]
        num_features = len(selected)
        if self.fitness_type == "original" and np.isfinite(best_gbest_val):
            accuracy = best_gbest_val + self.lambda_penalty * num_features
        else:
            accuracy, _ = self.calculate_objectives(best_selected_features)

        print("\nBest-selected binary vector:")
        print(best_selected_features)
        print(f"Best CV accuracy: {accuracy:.6f}")
        print(f"Best fitness/ranking score: {best_gbest_val:.6f}")
        print(f"Number of selected features: {num_features}")
        print("Selected feature names:")
        print(selected_feature_names)
        print(f"Fitness function used: {self.fitness_type}")
        print(f"KNN k: {self.knn_k}, CV folds: {self.cv}, lambda: {self.lambda_penalty}")
        print(f"ANOVA prefilter enabled: {self.use_anova_prefilter}")
        print(f"IMM rho: {self.rho}")

    def main(self, verbose=True):
        start_time = time.time()
        self.reset_ffe_counts()

        if self.remove_constant_features or self.use_anova_prefilter:
            self.preprocess_features()

        num_segments = min(15, self.D)
        orthogonal_table = OrthogonalArray(num_segments)
        pbest_val = np.full(self.Np, -np.inf, dtype=float)
        gbest_val = np.zeros(self.max_iter)
        v = np.zeros((self.D, self.Np))
        pbest = np.zeros((self.D, self.Np), dtype=int)
        gbest = np.zeros(self.D, dtype=int)
        gbest_store = np.zeros((self.D, self.max_iter), dtype=int)
        best_selected_features = np.zeros(self.D, dtype=int)
        best_gbest_val = -np.inf
        imm_count = max(1, self.Np // max(1, self.rho))
        conv_gen = 0

        pbest_val_avg_store = np.zeros(self.max_iter)
        fitness_avg_store = np.zeros(self.max_iter)
        features_index = list(range(self.D))

        x = self.initialization(features_index)

        for iteration in range(self.max_iter):
            r1 = np.random.rand(self.D, self.Np)
            r2 = np.random.rand(self.D, self.Np)

            if conv_gen >= self.max_conv_iter:
                break

            if iteration > 0:
                x, v = self.Evolve(
                    x, r1, r2, v, orthogonal_table, imm_count, ranking, num_segments, pbest, gbest
                )

            population_ffe_stage = (
                "population_initial"
                if iteration == 0
                else "population_after_movement"
            )
            fit = self.evaluate_population_fitness(
                x, stage=population_ffe_stage
            )
            ranking = np.argsort(fit)[::-1]

            if iteration > 0:
                improved = True
                while improved:
                    x, fit, ranking, improved = self.local_search(
                        x, orthogonal_table, features_index, ranking, num_segments, fit
                    )
                    if self.fitness_type == "gpsisf":
                        # GPSISF scores are population-relative. Refresh the population
                        # objectives/scores after an accepted elite change, but evaluate
                        # each member only once in this refresh.
                        if improved:
                            fit = self.evaluate_population_fitness(
                                x, stage="population_after_movement"
                            )
                            ranking = np.argsort(fit)[::-1]

            if iteration == 0:
                pbest_val = np.copy(fit)
                pbest = np.copy(x)
            else:
                pbest_val, pbest = self.update_p(fit, pbest_val, pbest, x)

            conv_gen, best_gbest_val, best_selected_features, gbest_val, gbest_store, gbest = self.update_g(
                best_selected_features,
                best_gbest_val,
                gbest_val,
                gbest_store,
                pbest_val,
                iteration,
                pbest,
                conv_gen,
            )

            if self.save_pareto:
                self.update_pareto_solutions(x, iteration)

            pbest_val_avg_store[iteration] = np.mean(pbest_val[np.isfinite(pbest_val)])
            fitness_avg_store[iteration] = np.mean(fit[np.isfinite(fit)])

            if verbose:
                self.show_iter(best_selected_features, best_gbest_val, conv_gen, iteration)

        self.show(best_selected_features, best_gbest_val)

        if self.save_pareto:
            self.generate_pareto_report()
            self.save_pareto_data()
            self.plot_pareto_front_2d()

        execution_time = time.time() - start_time
        print(f"execution_time: {execution_time:.4f} s")
        self.print_ffe_report()

        selected_final = self.selected_indices(best_selected_features)
        num_features = len(selected_final)
        if self.fitness_type == "original" and np.isfinite(best_gbest_val):
            accuracy = best_gbest_val + self.lambda_penalty * num_features
        else:
            accuracy, _ = self.calculate_objectives(best_selected_features)
        return {
            "best_selected_features": self.binarize(best_selected_features),
            "best_accuracy": accuracy,
            "best_fitness": best_gbest_val,
            "num_features": num_features,
            "selected_feature_names": self.feature_names[self.selected_indices(best_selected_features)],
            "execution_time": execution_time,
            "fitness_type": self.fitness_type,
            "knn_k": self.knn_k,
            "lambda_penalty": self.lambda_penalty,
            "use_anova_prefilter": self.use_anova_prefilter,
            "rho": self.rho,
            "ffe_counts": dict(self.ffe_counts),
            "total_search_ffe": self.total_search_ffe(),
            "cv_folds_per_ffe": self.cv,
            "internal_cv_fold_evaluations": self.total_search_ffe() * self.cv,
        }


# ----------------------------------------------------------------------
# Utility and command-line interface
# ----------------------------------------------------------------------
def load_csv_dataset(csv_file_path, encoding="utf-8"):
    data = pd.read_csv(csv_file_path, encoding=encoding)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    feature_names = data.columns[:-1].to_numpy()
    return X, y, feature_names


def get_dataset_name_from_path(csv_path):
    return Path(csv_path).stem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phoenix-OI: cross-phase OED/MED-guided PSO wrapper feature selection."
    )
    parser.add_argument("--data", type=str, required=True, help="Path to a local numeric CSV with the class label in its final column (download/prepare data separately; see README).")
    parser.add_argument("--encoding", type=str, default="utf-8", help="CSV file encoding.")
    parser.add_argument("--mode", choices=["original", "gpsisf"], default="original", help="Fitness mode.")
    parser.add_argument("--save_pareto", action="store_true", help="Save post-hoc Pareto analysis outputs.")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory.")
    parser.add_argument("--use_anova_prefilter", action="store_true", help="Enable optional ANOVA SelectKBest prefilter.")
    parser.add_argument("--no_constant_filter", action="store_true", help="Disable constant-feature removal.")
    parser.add_argument("--search_space", type=int, default=200, help="Feature count kept by optional ANOVA prefilter.")
    parser.add_argument("--Np", type=int, default=30, help="Swarm size.")
    parser.add_argument("--max_iter", type=int, default=30, help="Maximum iterations.")
    parser.add_argument("--max_conv_iter", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--tol", type=float, default=0.001, help="Convergence tolerance.")
    parser.add_argument("--rho", type=int, default=3, help="IMM divisor rho; floor(Np/rho) ranked particles use IMM, and rho=1 selects all particles.")
    parser.add_argument("--w", type=float, default=0.9, help="PSO inertia weight.")
    parser.add_argument("--c1", type=float, default=2.0, help="PSO personal coefficient.")
    parser.add_argument("--c2", type=float, default=2.0, help="PSO social coefficient.")
    parser.add_argument("--knn_k", type=int, default=3, help="K for KNN wrapper evaluator.")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--lambda_penalty", type=float, default=0.001, help="Feature-count penalty coefficient.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    X, y, feature_names = load_csv_dataset(args.data, encoding=args.encoding)
    dataset_name = get_dataset_name_from_path(args.data)

    print(f"Dataset: {dataset_name}")
    print(f"Mode: {args.mode}")
    print(f"KNN k={args.knn_k}, CV={args.cv}, lambda={args.lambda_penalty}")
    print(f"ANOVA prefilter enabled: {args.use_anova_prefilter}")
    print(f"IMM rho: {args.rho}")

    pso = GPPSO(
        feature=X,
        feature_names=feature_names,
        label=y,
        w=args.w,
        c1=args.c1,
        c2=args.c2,
        Np=args.Np,
        search_space=args.search_space,
        max_iter=args.max_iter,
        max_conv_iter=args.max_conv_iter,
        tol=args.tol,
        rho=args.rho,
        fitness_type=args.mode,
        lambda_penalty=args.lambda_penalty,
        knn_k=args.knn_k,
        cv=args.cv,
        use_anova_prefilter=args.use_anova_prefilter,
        remove_constant_features=not args.no_constant_filter,
        save_pareto=args.save_pareto,
        output_dir=args.output_dir,
        dataset_name=dataset_name,
        random_state=args.seed,
    )
    pso.main()


if __name__ == "__main__":
    main()
