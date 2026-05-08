"""
Part 2: Classification and Uncertainty Quantification

Este módulo implementa classificadores probabilísticos com quantificação de incerteza,
calibração e otimização de hiperparâmetros.
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from mapie.classification import SplitConformalClassifier


class ProbabilisticClassifier:
    """
    Classe para treino e avaliação de classificadores probabilísticos.
    """

    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.feature_encoders = {}

    def prepare_features(self, df: pd.DataFrame, fit_encoders: bool = False):
        """
        Prepara features numéricas e categóricas para classificação.

        Args:
            df: DataFrame com features
            fit_encoders: Se True, ajusta os encoders (apenas no treino)

        Returns:
            X_num: Features numéricas
            X_cat: Features categóricas codificadas
            y: Labels codificados
        """
        # Features numéricas
        num_features = [
            'ht_form_points', 'ht_form_goals_scored', 'ht_form_goals_conceded', 'ht_form_wins',
            'at_form_points', 'at_form_goals_scored', 'at_form_goals_conceded', 'at_form_wins',
            'ht_home_form_points', 'ht_home_form_goals_scored',
            'at_away_form_points', 'at_away_form_goals_scored',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins'
        ]

        # Features categóricas (ht_form/at_form são agora inteiros em num_features)
        cat_features = ['ht_attack', 'at_attack', 'h2h_dominance', 'odds_favorite']

        X_num = df[num_features].values
        X_cat_raw = df[cat_features].copy()

        # Codificar categóricas
        if fit_encoders:
            self.feature_encoders = {}
            for col in cat_features:
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_cat_raw[col] = encoder.fit_transform(X_cat_raw[[col]])
                self.feature_encoders[col] = encoder
        else:
            for col in cat_features:
                encoder = self.feature_encoders[col]
                X_cat_raw[col] = encoder.transform(X_cat_raw[[col]])

        X_cat = X_cat_raw.values

        # Label encoding
        if fit_encoders:
            y = self.label_encoder.fit_transform(df['result'])
        else:
            y = self.label_encoder.transform(df['result'])

        return X_num, X_cat, y

    def train_gaussian_nb(self, X_train, y_train, var_smoothing=1e-9):
        """
        Treina Gaussian Naive Bayes.

        Args:
            X_train: Features de treino (numéricas)
            y_train: Labels de treino
            var_smoothing: Parâmetro de suavização

        Returns:
            Modelo treinado
        """
        print("\nTreinando Gaussian Naive Bayes...")
        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(X_train, y_train)

        self.models['gaussian_nb'] = model
        print("Gaussian NB treinado com sucesso")
        return model

    def train_categorical_nb(self, X_train, y_train, alpha=1.0):
        """
        Treina Categorical Naive Bayes.

        Args:
            X_train: Features de treino (categóricas)
            y_train: Labels de treino
            alpha: Parâmetro de Laplace smoothing

        Returns:
            Modelo treinado
        """
        print("\nTreinando Categorical Naive Bayes...")
        model = CategoricalNB(alpha=alpha)
        model.fit(X_train, y_train)

        self.models['categorical_nb'] = model
        print("Categorical NB treinado com sucesso")
        return model

    def train_logistic_regression(self, X_train, y_train, C=1.0, solver='lbfgs'):
        """
        Treina Logistic Regression.

        Args:
            X_train: Features de treino
            y_train: Labels de treino
            C: Parâmetro de regularização
            solver: Algoritmo de otimização

        Returns:
            Modelo treinado
        """
        print("\nTreinando Logistic Regression...")
        model = LogisticRegression(C=C, solver=solver, max_iter=1000, multi_class='multinomial')
        model.fit(X_train, y_train)

        self.models['logistic_regression'] = model
        print("Logistic Regression treinada com sucesso")
        return model

    def conformal_prediction(self, X_train, y_train, X_test, y_test, alpha_levels=[0.05, 0.10, 0.20]):
        """
        Aplica Conformal Prediction para quantificação de incerteza.

        Args:
            X_train: Features de treino
            y_train: Labels de treino
            X_test: Features de teste
            y_test: Labels de teste
            alpha_levels: Níveis de significância

        Returns:
            Dicionário com resultados de cobertura
        """
        print("\n" + "="*70)
        print("Conformal Prediction")
        print("="*70)

        # Base estimator — treinado uma vez e reutilizado com prefit=True
        base_model = GaussianNB()
        split_idx = int(0.8 * len(X_train))
        X_train_fit, X_calib = X_train[:split_idx], X_train[split_idx:]
        y_train_fit, y_calib = y_train[:split_idx], y_train[split_idx:]
        base_model.fit(X_train_fit, y_train_fit)
        classes_list = list(base_model.classes_)

        results = {}

        for alpha in alpha_levels:
            conf_level = 1 - alpha
            print(f"\nConfiança: {conf_level:.2f} (alpha={alpha})")

            clf_cp = SplitConformalClassifier(
                estimator=base_model, confidence_level=conf_level,
                conformity_score='lac', prefit=True, random_state=42
            )
            clf_cp.conformalize(X_calib, y_calib)

            y_pred, ps_result = clf_cp.predict_set(X_test)
            ps = ps_result[:, :, 0]  # shape (n_samples, n_classes)

            coverage = np.mean([
                ps[i, classes_list.index(y_test[i])]
                for i in range(len(y_test))
            ])
            avg_set_size = float(ps.sum(axis=1).mean())

            print(f"  Cobertura empírica: {coverage:.3f}")
            print(f"  Tamanho médio do conjunto: {avg_set_size:.2f}")

            results[alpha] = {
                'coverage': coverage,
                'avg_set_size': avg_set_size,
                'predictions': y_pred,
                'prediction_sets': ps
            }

        return results

    def calibrate_model(self, model, X_calib, y_calib, method='isotonic'):
        """
        Calibra um modelo usando calibração isotónica ou sigmóide.

        Args:
            model: Modelo a calibrar
            X_calib: Features de calibração
            y_calib: Labels de calibração
            method: 'isotonic' ou 'sigmoid'

        Returns:
            Modelo calibrado
        """
        print(f"\nCalibrando modelo usando {method}...")
        calibrated = CalibratedClassifierCV(model, method=method, cv='prefit')
        calibrated.fit(X_calib, y_calib)

        print("Modelo calibrado com sucesso")
        return calibrated

    def bayesian_optimization(self, X_train, y_train, n_iter=30):
        """
        Otimiza hiperparâmetros usando Bayesian Optimization.

        Args:
            X_train: Features de treino
            y_train: Labels de treino
            n_iter: Número de iterações

        Returns:
            Melhor modelo encontrado
        """
        print("\n" + "="*70)
        print("Bayesian Hyperparameter Optimization")
        print("="*70)

        # Espaço de busca para Logistic Regression
        search_spaces = {
            'C': Real(0.001, 10.0, prior='log-uniform'),
            'solver': Categorical(['lbfgs', 'saga'])
        }

        tscv = TimeSeriesSplit(n_splits=5)

        # Configurar busca
        opt = BayesSearchCV(
            LogisticRegression(max_iter=1000, multi_class='multinomial'),
            search_spaces,
            n_iter=n_iter,
            cv=tscv,
            scoring='neg_log_loss',
            n_jobs=-1,
            random_state=42
        )

        print(f"Executando otimização Bayesiana com {n_iter} iterações...")
        opt.fit(X_train, y_train)

        print(f"\nMelhores hiperparâmetros: {opt.best_params_}")
        print(f"Melhor score (neg log loss): {opt.best_score_:.4f}")

        self.models['optimized_lr'] = opt.best_estimator_
        return opt.best_estimator_

    def evaluate(self, model, X_test, y_test, model_name="Model"):
        """
        Avalia um modelo no conjunto de teste.

        Args:
            model: Modelo treinado
            X_test: Features de teste
            y_test: Labels de teste
            model_name: Nome do modelo (para display)

        Returns:
            Dicionário com métricas
        """
        print(f"\n{'='*70}")
        print(f"Avaliação: {model_name}")
        print('='*70)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"Acurácia: {acc:.4f}")

        # Classification report
        print("\n" + classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))

        # Probabilistic metrics
        brier = brier_score_loss(y_test, y_proba, pos_label=1)
        logloss = log_loss(y_test, y_proba)

        print(f"Brier Score: {brier:.4f}")
        print(f"Log Loss: {logloss:.4f}")

        return {
            'accuracy': acc,
            'brier_score': brier,
            'log_loss': logloss,
            'predictions': y_pred,
            'probabilities': y_proba
        }


def run_classification_analysis(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Executa análise completa de classificação e incerteza.

    Args:
        train_data: Dados de treino
        test_data: Dados de teste

    Returns:
        Dicionário com todos os resultados
    """
    print("\n" + "="*70)
    print("PART 2: Classification and Uncertainty Quantification")
    print("="*70)

    classifier = ProbabilisticClassifier()

    # Preparar features
    print("\nPreparando features...")
    X_train_num, X_train_cat, y_train = classifier.prepare_features(train_data, fit_encoders=True)
    X_test_num, X_test_cat, y_test = classifier.prepare_features(test_data, fit_encoders=False)

    # Concatenar features numéricas e categóricas
    X_train_all = np.hstack([X_train_num, X_train_cat])
    X_test_all = np.hstack([X_test_num, X_test_cat])

    # 1. Treinar modelos Naive Bayes
    model_gnb = classifier.train_gaussian_nb(X_train_num, y_train)
    model_cnb = classifier.train_categorical_nb(X_train_cat, y_train)

    # 2. Avaliar Naive Bayes
    metrics_gnb = classifier.evaluate(model_gnb, X_test_num, y_test, "Gaussian Naive Bayes")
    metrics_cnb = classifier.evaluate(model_cnb, X_test_cat, y_test, "Categorical Naive Bayes")

    # 3. Conformal Prediction
    conformal_results = classifier.conformal_prediction(
        X_train_num, y_train, X_test_num, y_test,
        alpha_levels=[0.05, 0.10, 0.20]
    )

    # 4. Treinar Logistic Regression
    model_lr = classifier.train_logistic_regression(X_train_all, y_train)
    metrics_lr = classifier.evaluate(model_lr, X_test_all, y_test, "Logistic Regression")

    # 5. Calibração
    split_idx = int(0.8 * len(X_train_all))
    X_fit, X_calib = X_train_all[:split_idx], X_train_all[split_idx:]
    y_fit, y_calib = y_train[:split_idx], y_train[split_idx:]

    model_lr_uncalib = LogisticRegression(max_iter=1000, multi_class='multinomial')
    model_lr_uncalib.fit(X_fit, y_fit)

    model_lr_calib = classifier.calibrate_model(model_lr_uncalib, X_calib, y_calib, method='isotonic')
    metrics_lr_calib = classifier.evaluate(model_lr_calib, X_test_all, y_test, "LR Calibrated")

    # 6. Bayesian Optimization
    model_opt = classifier.bayesian_optimization(X_train_all, y_train, n_iter=20)
    metrics_opt = classifier.evaluate(model_opt, X_test_all, y_test, "Optimized LR")

    # Salvar melhor modelo
    np.save('data/best_model_proba_test.npy', metrics_opt['probabilities'])
    np.save('data/best_model_classes.npy', classifier.label_encoder.classes_)

    print("\nProbabilidades do melhor modelo salvas em data/")

    return {
        'classifier': classifier,
        'models': classifier.models,
        'metrics_gnb': metrics_gnb,
        'metrics_cnb': metrics_cnb,
        'metrics_lr': metrics_lr,
        'metrics_lr_calib': metrics_lr_calib,
        'metrics_opt': metrics_opt,
        'conformal_results': conformal_results
    }


if __name__ == "__main__":
    from data_pipeline import load_dataset, temporal_split

    df = load_dataset("../dataset/dataset.csv")
    train, test = temporal_split(df, split_year=2022)

    results = run_classification_analysis(train, test)
