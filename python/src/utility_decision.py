"""
Part 3: Utility Theory and Decision Making

Este módulo implementa análise de utilidade, comparação com odds do mercado,
e estratégias de decisão com reject option.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import accuracy_score


class UtilityAnalyzer:
    """
    Classe para análise de utilidade e tomada de decisão.
    """

    def __init__(self, test_data: pd.DataFrame, model_probabilities: np.ndarray, class_names: np.ndarray):
        """
        Inicializa o analisador de utilidade.

        Args:
            test_data: DataFrame de teste com odds
            model_probabilities: Probabilidades previstas pelo modelo (shape: n_samples x n_classes)
            class_names: Nomes das classes (e.g., ['A', 'D', 'H'])
        """
        self.test_data = test_data.copy()
        self.model_proba = model_probabilities
        self.class_names = class_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    def extract_odds_probabilities(self) -> pd.DataFrame:
        """
        Converte odds em probabilidades implícitas normalizadas.

        Returns:
            DataFrame com probabilidades implícitas
        """
        print("\n" + "="*70)
        print("Extração de Probabilidades Implícitas das Odds")
        print("="*70)

        # Probabilidades brutas (inverso das odds)
        p_h_raw = 1 / self.test_data['h_odds']
        p_d_raw = 1 / self.test_data['d_odds']
        p_a_raw = 1 / self.test_data['a_odds']

        # Normalizar (remover overround)
        total = p_h_raw + p_d_raw + p_a_raw

        p_h = p_h_raw / total
        p_d = p_d_raw / total
        p_a = p_a_raw / total

        # Margem do bookmaker
        overround = (total - 1) * 100

        print(f"Margem média do bookmaker: {overround.mean():.2f}%")
        print(f"\nProbabilidades implícitas médias:")
        print(f"  P(H) = {p_h.mean():.3f}")
        print(f"  P(D) = {p_d.mean():.3f}")
        print(f"  P(A) = {p_a.mean():.3f}")

        # Criar DataFrame
        odds_proba = pd.DataFrame({
            'p_H': p_h,
            'p_D': p_d,
            'p_A': p_a,
            'overround': overround
        })

        self.odds_proba = odds_proba
        return odds_proba

    def calculate_disagreement(self) -> pd.DataFrame:
        """
        Calcula a discordância entre modelo e mercado.

        Returns:
            DataFrame com métricas de discordância
        """
        print("\n" + "="*70)
        print("Análise de Discordância Modelo vs. Mercado")
        print("="*70)

        # Extrair probabilidades do modelo (assumindo ordem A, D, H)
        idx_A = self.class_to_idx.get('A', 0)
        idx_D = self.class_to_idx.get('D', 1)
        idx_H = self.class_to_idx.get('H', 2)

        model_p_h = self.model_proba[:, idx_H]
        model_p_d = self.model_proba[:, idx_D]
        model_p_a = self.model_proba[:, idx_A]

        # Discordância: max diferença absoluta entre classes
        disagreement = np.maximum(
            np.abs(model_p_h - self.odds_proba['p_H']),
            np.maximum(
                np.abs(model_p_d - self.odds_proba['p_D']),
                np.abs(model_p_a - self.odds_proba['p_A'])
            )
        )

        # KL-divergence (modelo || odds)
        kl_div = []
        for i in range(len(self.odds_proba)):
            p_model = [model_p_a[i], model_p_d[i], model_p_h[i]]
            p_odds = [self.odds_proba.iloc[i]['p_A'],
                     self.odds_proba.iloc[i]['p_D'],
                     self.odds_proba.iloc[i]['p_H']]

            kl = entropy(p_model, p_odds)
            kl_div.append(kl)

        disagreement_df = pd.DataFrame({
            'disagreement': disagreement,
            'kl_divergence': kl_div
        })

        print(f"Discordância média: {disagreement.mean():.3f}")
        print(f"Discordância mediana: {np.median(disagreement):.3f}")
        print(f"KL-divergência média: {np.mean(kl_div):.3f}")
        print(f"\nJogos com alta discordância (>0.15): {(disagreement > 0.15).sum()} ({100*(disagreement > 0.15).mean():.1f}%)")
        print(f"Jogos com discordância muito alta (>0.25): {(disagreement > 0.25).sum()} ({100*(disagreement > 0.25).mean():.1f}%)")

        self.disagreement_df = disagreement_df
        return disagreement_df

    def baseline_odds_accuracy(self) -> float:
        """
        Calcula acurácia baseline usando apenas as odds.

        Returns:
            Acurácia das odds
        """
        # Prever classe com maior probabilidade implícita
        odds_matrix = self.odds_proba[['p_A', 'p_D', 'p_H']].values
        odds_predictions = np.argmax(odds_matrix, axis=1)

        # Mapear índices para classes
        idx_to_class = {0: 'A', 1: 'D', 2: 'H'}
        odds_pred_labels = [idx_to_class[idx] for idx in odds_predictions]

        # Labels verdadeiros
        true_labels = self.test_data['result'].values

        acc = accuracy_score(true_labels, odds_pred_labels)
        print(f"\nAcurácia baseline (odds do mercado): {acc:.4f}")

        return acc

    def reject_option_strategy(self, confidence_thresholds=None) -> dict:
        """
        Implementa estratégia de reject option baseada em confiança.

        Args:
            confidence_thresholds: Lista de limiares de confiança

        Returns:
            Dicionário com resultados para cada limiar
        """
        print("\n" + "="*70)
        print("Reject Option Strategy")
        print("="*70)

        if confidence_thresholds is None:
            confidence_thresholds = np.arange(0.35, 0.71, 0.05)

        # Prever classe com máxima probabilidade
        model_pred_idx = np.argmax(self.model_proba, axis=1)
        model_confidence = np.max(self.model_proba, axis=1)

        # Prever usando odds
        odds_matrix = self.odds_proba[['p_A', 'p_D', 'p_H']].values
        odds_pred_idx = np.argmax(odds_matrix, axis=1)

        # Mapear índices para classes
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        model_pred_labels = np.array([idx_to_class[idx] for idx in model_pred_idx])
        odds_pred_labels = np.array([idx_to_class.get(idx, 'H') for idx in odds_pred_idx])

        true_labels = self.test_data['result'].values

        results = {}

        for threshold in confidence_thresholds:
            # Casos onde o modelo tem confiança suficiente
            accept_mask = model_confidence >= threshold

            # Estratégia pura: usar modelo quando aceite, abster-se quando rejeitado
            accepted_preds = model_pred_labels[accept_mask]
            accepted_true = true_labels[accept_mask]

            if len(accepted_preds) > 0:
                acc_conditional = accuracy_score(accepted_true, accepted_preds)
            else:
                acc_conditional = 0.0

            # Estratégia híbrida: usar odds quando modelo rejeita
            hybrid_preds = np.where(accept_mask, model_pred_labels, odds_pred_labels)
            acc_hybrid = accuracy_score(true_labels, hybrid_preds)

            coverage = accept_mask.mean()

            results[threshold] = {
                'coverage': coverage,
                'accuracy_conditional': acc_conditional,
                'accuracy_hybrid': acc_hybrid,
                'n_accepted': accept_mask.sum(),
                'n_rejected': (~accept_mask).sum()
            }

        # Exibir alguns resultados
        print(f"\n{'Threshold':<12} {'Coverage':<12} {'Acc (Condicional)':<20} {'Acc (Híbrida)':<15}")
        print("-"*70)

        for thresh in [0.40, 0.44, 0.50, 0.60]:
            if thresh in results:
                r = results[thresh]
                print(f"{thresh:<12.2f} {r['coverage']:<12.2f} {r['accuracy_conditional']:<20.3f} {r['accuracy_hybrid']:<15.3f}")

        return results

    def betting_simulation(self, confidence_threshold=0.458, initial_bankroll=100.0) -> dict:
        """
        Simula estratégia de apostas baseada no Princípio da Utilidade Máxima Esperada (MEU).

        Condições para apostar:
          1. Confiança do modelo >= confidence_threshold (limiar da Reject Option)
          2. EU > 0 para pelo menos um resultado, i.e., p_modelo > 1/odd (EV positivo)

        Aposta no resultado com maior EU positiva. Stake fixo de €1
        (função de utilidade linear).

        Args:
            confidence_threshold: Limiar de confiança mínima (0.458, da análise Reject Option)
            initial_bankroll: Bankroll inicial (€)

        Returns:
            Dicionário com resultados da simulação
        """
        print("\n" + "="*70)
        print("Betting Simulation (MEU — Maximum Expected Utility)")
        print("="*70)

        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        bets_placed = 0
        bets_won = 0
        stake = 1.0

        model_confidence = np.max(self.model_proba, axis=1)

        idx_A = self.class_to_idx.get('A', 0)
        idx_D = self.class_to_idx.get('D', 1)
        idx_H = self.class_to_idx.get('H', 2)

        argmax_pred = [
            self.class_names[np.argmax(self.model_proba[i])]
            for i in range(len(self.test_data))
        ]

        for i in range(len(self.test_data)):
            # Condição 1: confiança mínima (da Reject Option)
            if model_confidence[i] < confidence_threshold:
                bankroll_history.append(bankroll)
                continue

            pred = argmax_pred[i]
            idx_pred = self.class_to_idx[pred]
            p   = self.model_proba[i, idx_pred]
            odd_key = {'H': 'h_odds', 'D': 'd_odds', 'A': 'a_odds'}[pred]
            odd = self.test_data.iloc[i][odd_key]

            # Condição 2: EU > 0 para o resultado previsto (p_modelo > 1/odd)
            eu = p * (odd - 1) - (1 - p)
            if eu <= 0:
                bankroll_history.append(bankroll)
                continue

            bets_placed += 1
            bankroll -= stake
            true_result = self.test_data.iloc[i]['result']

            if true_result == pred:
                bankroll += stake * odd
                bets_won += 1

            bankroll_history.append(bankroll)

        final_pnl = bankroll - initial_bankroll
        roi = (final_pnl / bets_placed) * 100 if bets_placed > 0 else 0.0

        print(f"\nBankroll inicial: €{initial_bankroll:.2f}")
        print(f"Bankroll final:   €{bankroll:.2f}")
        print(f"P&L:              €{final_pnl:.2f}")
        print(f"\nApostas colocadas: {bets_placed}")
        print(f"Apostas ganhas:    {bets_won}")
        print(f"Taxa de vitória:   {100*bets_won/bets_placed if bets_placed > 0 else 0:.1f}%")
        print(f"ROI por aposta:    {roi:.2f}%")

        return {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': bankroll,
            'pnl': final_pnl,
            'bets_placed': bets_placed,
            'bets_won': bets_won,
            'win_rate': bets_won / bets_placed if bets_placed > 0 else 0,
            'roi': roi,
            'bankroll_history': bankroll_history
        }


def run_utility_analysis(test_data: pd.DataFrame, model_proba_path: str, classes_path: str):
    """
    Executa análise completa de utilidade e decisão.

    Args:
        test_data: Dados de teste
        model_proba_path: Caminho para probabilidades do modelo
        classes_path: Caminho para classes do modelo

    Returns:
        Dicionário com todos os resultados
    """
    print("\n" + "="*70)
    print("PART 3: Utility Theory and Decision Making")
    print("="*70)

    # Carregar probabilidades do melhor modelo
    model_proba = np.load(model_proba_path)
    class_names = np.load(classes_path)

    print(f"\nCarregadas probabilidades: shape {model_proba.shape}")
    print(f"Classes: {class_names}")

    # Inicializar analisador
    analyzer = UtilityAnalyzer(test_data, model_proba, class_names)

    # 1. Extrair probabilidades das odds
    odds_proba = analyzer.extract_odds_probabilities()

    # 2. Acurácia baseline das odds
    odds_accuracy = analyzer.baseline_odds_accuracy()

    # 3. Calcular discordância
    disagreement = analyzer.calculate_disagreement()

    # 4. Estratégia de Reject Option
    reject_results = analyzer.reject_option_strategy()

    # 5. Simulação de apostas (MEU: EU > 0 + confiança >= 0.458)
    betting_results = analyzer.betting_simulation(
        confidence_threshold=0.458,
        initial_bankroll=100.0
    )

    return {
        'analyzer': analyzer,
        'odds_proba': odds_proba,
        'odds_accuracy': odds_accuracy,
        'disagreement': disagreement,
        'reject_results': reject_results,
        'betting_results': betting_results
    }


if __name__ == "__main__":
    from data_pipeline import load_dataset, temporal_split

    df = load_dataset("../dataset/dataset.csv")
    train, test = temporal_split(df, split_year=2022)

    results = run_utility_analysis(
        test,
        "../data/best_model_proba_test.npy",
        "../data/best_model_classes.npy"
    )
