import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import re
from typing import Dict, List


class MLResearchPredictor:
    """
    ML-based research paper analysis using 5-model ensemble:
    1. Random Forest
    2. Gradient Boosting
    3. Ridge Regression / Logistic Regression
    4. Support Vector Machine (SVR / SVC)
    5. K-Nearest Neighbors (KNN)

    Predicts:
    - Reproducibility Score (regression, 1-10 scale)
    - Difficulty Level (classification: Beginner / Intermediate / Advanced)
    """

    def __init__(self):
        self.scaler = StandardScaler()

        # ── 5 Reproducibility Score Models (Regression: 1-10) ──
        self.repro_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.repro_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.repro_ridge = Ridge(alpha=1.0)
        self.repro_svr = SVR(kernel="rbf", C=1.0)
        self.repro_knn = KNeighborsRegressor(n_neighbors=5)

        # ── 5 Difficulty Level Models (Classification: 0/1/2) ──
        self.diff_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.diff_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.diff_lr = LogisticRegression(max_iter=1000, random_state=42)
        self.diff_svc = SVC(kernel="rbf", probability=True, random_state=42)
        self.diff_knn = KNeighborsClassifier(n_neighbors=5)

        self.is_trained = False
        self._train_with_synthetic_data()

    def extract_features(self, text: str, equations: List[str],
                         sections: Dict, references: List[str]) -> np.ndarray:
        """Extract 12 numerical features from a parsed paper."""
        words = text.split()
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        unique_words = set(w.lower() for w in words)

        features = [
            len(equations),                                                    # 1. equation_count
            len(text) // 250,                                                  # 2. page_estimate
            len(references),                                                   # 3. reference_count
            np.mean([len(s.split()) for s in sentences]) if sentences else 0,  # 4. avg_sentence_length
            len(unique_words) / max(len(words), 1),                            # 5. unique_word_ratio
            len(sections),                                                     # 6. section_count
            text.lower().count("figure") + text.lower().count("fig."),         # 7. figure_mentions
            text.lower().count("table"),                                       # 8. table_mentions
            text.lower().count("dataset") + text.lower().count("data set"),    # 9. dataset_mentions
            text.lower().count("github") + text.lower().count("code"),         # 10. code_mentions
            text.lower().count("hyperparameter") + text.lower().count("learning rate"),  # 11. hyperparam_mentions
            len(words),                                                        # 12. total_word_count
        ]
        return np.array(features).reshape(1, -1)

    def _train_with_synthetic_data(self):
        """Train all 10 models (5 regression + 5 classification) with synthetic data."""
        np.random.seed(42)
        n_samples = 300

        # Generate synthetic feature data simulating diverse paper types
        X = np.column_stack([
            np.random.randint(0, 30, n_samples),       # equation_count
            np.random.randint(4, 40, n_samples),        # page_estimate
            np.random.randint(5, 80, n_samples),        # reference_count
            np.random.uniform(10, 35, n_samples),       # avg_sentence_length
            np.random.uniform(0.15, 0.65, n_samples),   # unique_word_ratio
            np.random.randint(3, 15, n_samples),        # section_count
            np.random.randint(0, 20, n_samples),        # figure_mentions
            np.random.randint(0, 15, n_samples),        # table_mentions
            np.random.randint(0, 10, n_samples),        # dataset_mentions
            np.random.randint(0, 8, n_samples),         # code_mentions
            np.random.randint(0, 12, n_samples),        # hyperparam_mentions
            np.random.randint(1000, 15000, n_samples),  # total_word_count
        ])

        # ── Reproducibility target (1-10 scale) ──
        # Higher when code, data, and hyperparams are well-documented
        y_repro = np.clip(
            3.0
            + 0.15 * X[:, 9]    # code_mentions boost
            + 0.12 * X[:, 8]    # dataset_mentions boost
            + 0.10 * X[:, 10]   # hyperparam_mentions boost
            + 0.05 * X[:, 7]    # table_mentions boost
            - 0.03 * X[:, 0]    # equation_count penalty
            - 0.02 * X[:, 1]    # longer papers harder
            + np.random.normal(0, 0.5, n_samples),
            1, 10
        )

        # ── Difficulty target (0=Beginner, 1=Intermediate, 2=Advanced) ──
        difficulty_score = (
            0.08 * X[:, 0]     # equation_count
            + 0.05 * X[:, 3]   # avg_sentence_length
            - 1.5 * X[:, 4]    # unique_word_ratio (higher = simpler)
            + 0.03 * X[:, 2]   # reference_count
            + np.random.normal(0, 0.3, n_samples)
        )
        thresholds = [np.percentile(difficulty_score, 33), np.percentile(difficulty_score, 66)]
        y_diff = np.digitize(difficulty_score, bins=thresholds)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # ── Train 5 Reproducibility Models ──
        self.repro_rf.fit(X_scaled, y_repro)
        self.repro_gb.fit(X_scaled, y_repro)
        self.repro_ridge.fit(X_scaled, y_repro)
        self.repro_svr.fit(X_scaled, y_repro)
        self.repro_knn.fit(X_scaled, y_repro)

        # ── Train 5 Difficulty Models ──
        self.diff_rf.fit(X_scaled, y_diff)
        self.diff_gb.fit(X_scaled, y_diff)
        self.diff_lr.fit(X_scaled, y_diff)
        self.diff_svc.fit(X_scaled, y_diff)
        self.diff_knn.fit(X_scaled, y_diff)

        self.is_trained = True

    def predict_reproducibility(self, text: str, equations: List[str],
                                 sections: Dict, references: List[str]) -> Dict:
        """Predict reproducibility score using ensemble of 5 models."""
        features = self.extract_features(text, equations, sections, references)
        features_scaled = self.scaler.transform(features)

        # Get predictions from all 5 models
        pred_rf = float(np.clip(self.repro_rf.predict(features_scaled)[0], 1, 10))
        pred_gb = float(np.clip(self.repro_gb.predict(features_scaled)[0], 1, 10))
        pred_ridge = float(np.clip(self.repro_ridge.predict(features_scaled)[0], 1, 10))
        pred_svr = float(np.clip(self.repro_svr.predict(features_scaled)[0], 1, 10))
        pred_knn = float(np.clip(self.repro_knn.predict(features_scaled)[0], 1, 10))

        # Ensemble average
        all_preds = [pred_rf, pred_gb, pred_ridge, pred_svr, pred_knn]
        ensemble_score = round(sum(all_preds) / len(all_preds), 1)

        return {
            "ensemble_score": ensemble_score,
            "model_predictions": {
                "Random Forest": round(pred_rf, 1),
                "Gradient Boosting": round(pred_gb, 1),
                "Ridge Regression": round(pred_ridge, 1),
                "SVM (SVR)": round(pred_svr, 1),
                "KNN": round(pred_knn, 1),
            },
            "interpretation": self._interpret_repro(ensemble_score),
        }

    def predict_difficulty(self, text: str, equations: List[str],
                           sections: Dict, references: List[str]) -> Dict:
        """Predict difficulty level using ensemble of 5 models."""
        features = self.extract_features(text, equations, sections, references)
        features_scaled = self.scaler.transform(features)

        labels = ["Beginner", "Intermediate", "Advanced"]

        # Get predictions from all 5 models
        pred_rf = int(self.diff_rf.predict(features_scaled)[0])
        pred_gb = int(self.diff_gb.predict(features_scaled)[0])
        pred_lr = int(self.diff_lr.predict(features_scaled)[0])
        pred_svc = int(self.diff_svc.predict(features_scaled)[0])
        pred_knn = int(self.diff_knn.predict(features_scaled)[0])

        # Majority vote ensemble
        votes = [pred_rf, pred_gb, pred_lr, pred_svc, pred_knn]
        ensemble_pred = max(set(votes), key=votes.count)

        # Get probability estimates from each model
        prob_rf = self.diff_rf.predict_proba(features_scaled)[0].tolist()
        prob_gb = self.diff_gb.predict_proba(features_scaled)[0].tolist()
        prob_lr = self.diff_lr.predict_proba(features_scaled)[0].tolist()
        prob_svc = self.diff_svc.predict_proba(features_scaled)[0].tolist()
        prob_knn = self.diff_knn.predict_proba(features_scaled)[0].tolist()

        # Average probabilities across all 5 models
        avg_probs = [
            (p1 + p2 + p3 + p4 + p5) / 5
            for p1, p2, p3, p4, p5 in zip(prob_rf, prob_gb, prob_lr, prob_svc, prob_knn)
        ]

        return {
            "ensemble_level": labels[ensemble_pred],
            "model_predictions": {
                "Random Forest": labels[pred_rf],
                "Gradient Boosting": labels[pred_gb],
                "Logistic Regression": labels[pred_lr],
                "SVM (SVC)": labels[pred_svc],
                "KNN": labels[pred_knn],
            },
            "confidence": {labels[i]: round(avg_probs[i] * 100, 1) for i in range(len(labels))},
        }

    def _interpret_repro(self, score: float) -> str:
        """Provide human-readable interpretation of reproducibility score."""
        if score >= 8:
            return "🟢 Highly Reproducible — Clear methodology with available code/data."
        elif score >= 6:
            return "🟡 Moderately Reproducible — Some details may need clarification."
        elif score >= 4:
            return "🟠 Challenging — Missing key implementation details."
        else:
            return "🔴 Difficult to Reproduce — Significant information gaps."

    def get_feature_importance(self) -> Dict:
        """Get feature importance from the Random Forest models."""
        feature_names = [
            "Equation Count", "Page Estimate", "Reference Count",
            "Avg Sentence Length", "Unique Word Ratio", "Section Count",
            "Figure Mentions", "Table Mentions", "Dataset Mentions",
            "Code Mentions", "Hyperparameter Mentions", "Total Words",
        ]
        repro_imp = self.repro_rf.feature_importances_
        diff_imp = self.diff_rf.feature_importances_

        return {
            "reproducibility": {n: round(float(v), 4) for n, v in zip(feature_names, repro_imp)},
            "difficulty": {n: round(float(v), 4) for n, v in zip(feature_names, diff_imp)},
        }
