#!/usr/bin/env python3
"""
Main Script - Projeto IAP: Previsão Probabilística de Futebol
Versão Python Modular

Este script executa todo o pipeline do projeto:
- Part 0: Data Pipeline & EDA
- Part 1: Bayesian Networks
- Part 2: Classification & Uncertainty
- Part 3: Utility & Decision

Uso:
    python main.py [--skip-part PART_NUMBER] [--only-part PART_NUMBER]

Exemplos:
    python main.py                    # Executar todas as partes
    python main.py --only-part 2      # Executar apenas Part 2
    python main.py --skip-part 1      # Pular Part 1
"""

import argparse
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_pipeline import prepare_dataset, temporal_split
from src.bayesian_networks import run_bayesian_networks_analysis
from src.classification import run_classification_analysis
from src.utility_decision import run_utility_analysis


def print_header(title: str):
    """Imprime cabeçalho formatado."""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)


def run_part0(dataset_path: str, output_path: str = None) -> tuple:
    """
    Executa Part 0: Data Pipeline and EDA.

    Returns:
        (train_data, test_data)
    """
    print_header("PART 0: DATA PIPELINE AND EXPLORATORY DATA ANALYSIS")

    # Preparar dataset
    df = prepare_dataset(dataset_path, output_path)

    # Split temporal
    train, test = temporal_split(df, split_year=2022)

    print("\n✓ Part 0 concluída com sucesso")

    return train, test


def run_part1(train_data, test_data) -> dict:
    """
    Executa Part 1: Bayesian Networks.

    Returns:
        Dicionário com resultados da análise
    """
    print_header("PART 1: BAYESIAN NETWORKS")

    results = run_bayesian_networks_analysis(train_data, test_data)

    print("\n✓ Part 1 concluída com sucesso")

    return results


def run_part2(train_data, test_data) -> dict:
    """
    Executa Part 2: Classification and Uncertainty Quantification.

    Returns:
        Dicionário com resultados da classificação
    """
    print_header("PART 2: CLASSIFICATION AND UNCERTAINTY QUANTIFICATION")

    results = run_classification_analysis(train_data, test_data)

    print("\n✓ Part 2 concluída com sucesso")

    return results


def run_part3(test_data, proba_path: str, classes_path: str) -> dict:
    """
    Executa Part 3: Utility Theory and Decision Making.

    Returns:
        Dicionário com resultados da análise de utilidade
    """
    print_header("PART 3: UTILITY THEORY AND DECISION MAKING")

    results = run_utility_analysis(test_data, proba_path, classes_path)

    print("\n✓ Part 3 concluída com sucesso")

    return results


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Projeto IAP - Previsão Probabilística de Futebol'
    )
    parser.add_argument('--skip-part', type=int, choices=[0, 1, 2, 3],
                       help='Pular uma parte específica (0-3)')
    parser.add_argument('--only-part', type=int, choices=[0, 1, 2, 3],
                       help='Executar apenas uma parte específica (0-3)')
    parser.add_argument('--dataset', type=str, default='data/dataset.csv',
                       help='Caminho para o dataset')

    args = parser.parse_args()

    print_header("PROJETO IAP 2025/2026 - PREVISÃO PROBABILÍSTICA DE FUTEBOL")
    print(f"\nAutor: Duarte Prazeres (2023214962)")
    print(f"Dataset: {args.dataset}")

    # Determinar quais partes executar
    parts_to_run = set([0, 1, 2, 3])

    if args.only_part is not None:
        parts_to_run = {args.only_part}
        print(f"\nModo: Executar apenas Part {args.only_part}")
    elif args.skip_part is not None:
        parts_to_run.remove(args.skip_part)
        print(f"\nModo: Pular Part {args.skip_part}")
    else:
        print("\nModo: Executar todas as partes")

    # Inicializar variáveis
    train_data = None
    test_data = None
    results = {}

    try:
        # Part 0: Data Pipeline (sempre necessário se outras partes forem executadas)
        if 0 in parts_to_run or len(parts_to_run & {1, 2, 3}) > 0:
            train_data, test_data = run_part0(
                args.dataset,
                output_path='data/dataset_processed.csv'
            )
            results['part0'] = {'train': train_data, 'test': test_data}

        # Part 1: Bayesian Networks
        if 1 in parts_to_run:
            if train_data is None:
                print("\n⚠ Erro: Part 0 necessária antes de Part 1")
                return

            results['part1'] = run_part1(train_data, test_data)

        # Part 2: Classification
        if 2 in parts_to_run:
            if train_data is None:
                print("\n⚠ Erro: Part 0 necessária antes de Part 2")
                return

            results['part2'] = run_part2(train_data, test_data)

        # Part 3: Utility & Decision
        if 3 in parts_to_run:
            if test_data is None:
                print("\n⚠ Erro: Part 0 necessária antes de Part 3")
                return

            # Verificar se probabilidades foram geradas
            proba_path = 'data/best_model_proba_test.npy'
            classes_path = 'data/best_model_classes.npy'

            if not Path(proba_path).exists():
                print(f"\n⚠ Erro: {proba_path} não encontrado")
                print("Execute Part 2 primeiro para gerar as probabilidades")
                return

            results['part3'] = run_part3(test_data, proba_path, classes_path)

        # Sumário final
        print_header("SUMÁRIO FINAL")

        if 'part1' in results:
            metrics_ke = results['part1']['metrics_ke']
            metrics_hc = results['part1']['metrics_hc']
            print(f"\n[Part 1] Redes Bayesianas:")
            print(f"  - Modelo KE: Acurácia = {metrics_ke['accuracy']:.4f}")
            print(f"  - Modelo HC: Acurácia = {metrics_hc['accuracy']:.4f}")

        if 'part2' in results:
            metrics_opt = results['part2']['metrics_opt']
            print(f"\n[Part 2] Classificação Probabilística:")
            print(f"  - Melhor modelo: Acurácia = {metrics_opt['accuracy']:.4f}")
            print(f"  - Log Loss = {metrics_opt['log_loss']:.4f}")
            print(f"  - Brier Score = {metrics_opt['brier_score']:.4f}")

        if 'part3' in results:
            odds_acc = results['part3']['odds_accuracy']
            betting = results['part3']['betting_results']
            print(f"\n[Part 3] Utilidade e Decisão:")
            print(f"  - Acurácia odds do mercado: {odds_acc:.4f}")
            print(f"  - Simulação de apostas: P&L = €{betting['pnl']:.2f}")
            print(f"  - Apostas colocadas: {betting['bets_placed']}")
            print(f"  - Taxa de vitória: {100*betting['win_rate']:.1f}%")

        print("\n" + "="*80)
        print("✓ Pipeline completo executado com sucesso!")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
