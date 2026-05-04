"""
Part 0: Data Pipeline and Feature Engineering

Este módulo contém funções para carregamento, limpeza e engenharia de features
para o dataset de futebol português.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Carrega o dataset de jogos de futebol.

    Args:
        file_path: Caminho para o ficheiro CSV

    Returns:
        DataFrame com os dados carregados
    """
    df = pd.read_csv(file_path, parse_dates=['date'])
    print(f"Dataset carregado: {len(df)} jogos")
    print(f"Período: {df['date'].min()} a {df['date'].max()}")
    return df


def compute_form_features(match_data: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """
    Calcula features de forma recente (últimos N jogos) para cada equipa.

    Args:
        match_data: DataFrame com dados dos jogos
        n_matches: Número de jogos a considerar para forma recente

    Returns:
        DataFrame com features de forma adicionadas
    """
    features = {
        'ht_form_points': [],
        'ht_form_goals_scored': [],
        'ht_form_goals_conceded': [],
        'ht_form_wins': [],
        'at_form_points': [],
        'at_form_goals_scored': [],
        'at_form_goals_conceded': [],
        'at_form_wins': [],
        'ht_home_form_points': [],
        'ht_home_form_goals_scored': [],
        'at_away_form_points': [],
        'at_away_form_goals_scored': [],
        'h2h_home_wins': [],
        'h2h_draws': [],
        'h2h_away_wins': [],
    }

    for idx, row in match_data.iterrows():
        home_team = row['ht']
        away_team = row['at']

        past = match_data.iloc[:idx]

        # Features de forma geral
        ht_past = past[(past['ht'] == home_team) | (past['at'] == home_team)].tail(n_matches)
        at_past = past[(past['ht'] == away_team) | (past['at'] == away_team)].tail(n_matches)

        features['ht_form_points'].append(calc_team_points(ht_past, home_team))
        features['ht_form_goals_scored'].append(calc_team_goals_scored(ht_past, home_team))
        features['ht_form_goals_conceded'].append(calc_team_goals_conceded(ht_past, home_team))
        features['ht_form_wins'].append(calc_team_wins(ht_past, home_team))

        features['at_form_points'].append(calc_team_points(at_past, away_team))
        features['at_form_goals_scored'].append(calc_team_goals_scored(at_past, away_team))
        features['at_form_goals_conceded'].append(calc_team_goals_conceded(at_past, away_team))
        features['at_form_wins'].append(calc_team_wins(at_past, away_team))

        # Features de forma casa/fora
        ht_home_past = past[past['ht'] == home_team].tail(n_matches)
        at_away_past = past[past['at'] == away_team].tail(n_matches)

        features['ht_home_form_points'].append(calc_home_points(ht_home_past))
        features['ht_home_form_goals_scored'].append(
            ht_home_past['hg'].sum() if len(ht_home_past) > 0 else np.nan
        )

        features['at_away_form_points'].append(calc_away_points(at_away_past))
        features['at_away_form_goals_scored'].append(
            at_away_past['ag'].sum() if len(at_away_past) > 0 else np.nan
        )

        # Features head-to-head
        h2h = past[
            ((past['ht'] == home_team) & (past['at'] == away_team)) |
            ((past['ht'] == away_team) & (past['at'] == home_team))
        ]

        h2h_hw, h2h_d, h2h_aw = calc_h2h(h2h, home_team, away_team)
        features['h2h_home_wins'].append(h2h_hw)
        features['h2h_draws'].append(h2h_d)
        features['h2h_away_wins'].append(h2h_aw)

    for col, values in features.items():
        match_data[col] = values

    return match_data


def calc_team_points(df: pd.DataFrame, team: str) -> float:
    """Calcula pontos totais de uma equipa."""
    if len(df) == 0:
        return np.nan

    points = 0
    for _, row in df.iterrows():
        if row['ht'] == team:
            if row['result'] == 'H':
                points += 3
            elif row['result'] == 'D':
                points += 1
        else:
            if row['result'] == 'A':
                points += 3
            elif row['result'] == 'D':
                points += 1
    return points


def calc_team_wins(df: pd.DataFrame, team: str) -> float:
    """Calcula número de vitórias de uma equipa."""
    if len(df) == 0:
        return np.nan

    wins = 0
    for _, row in df.iterrows():
        if row['ht'] == team and row['result'] == 'H':
            wins += 1
        elif row['at'] == team and row['result'] == 'A':
            wins += 1
    return wins


def calc_team_goals_scored(df: pd.DataFrame, team: str) -> float:
    """Calcula golos marcados por uma equipa."""
    if len(df) == 0:
        return np.nan

    goals = 0
    for _, row in df.iterrows():
        if row['ht'] == team:
            goals += row['hg']
        else:
            goals += row['ag']
    return goals


def calc_team_goals_conceded(df: pd.DataFrame, team: str) -> float:
    """Calcula golos sofridos por uma equipa."""
    if len(df) == 0:
        return np.nan

    goals = 0
    for _, row in df.iterrows():
        if row['ht'] == team:
            goals += row['ag']
        else:
            goals += row['hg']
    return goals


def calc_home_points(df: pd.DataFrame) -> float:
    """Calcula pontos em jogos em casa."""
    if len(df) == 0:
        return np.nan
    return (df['result'] == 'H').sum() * 3 + (df['result'] == 'D').sum()


def calc_away_points(df: pd.DataFrame) -> float:
    """Calcula pontos em jogos fora."""
    if len(df) == 0:
        return np.nan
    return (df['result'] == 'A').sum() * 3 + (df['result'] == 'D').sum()


def calc_h2h(df: pd.DataFrame, home_team: str, away_team: str) -> tuple:
    """
    Calcula estatísticas head-to-head entre duas equipas.

    Returns:
        Tupla (vitórias_casa, empates, vitórias_fora)
    """
    if len(df) == 0:
        return np.nan, np.nan, np.nan

    home_wins, draws, away_wins = 0, 0, 0

    for _, row in df.iterrows():
        if row['result'] == 'D':
            draws += 1
        elif row['ht'] == home_team:
            if row['result'] == 'H':
                home_wins += 1
            else:
                away_wins += 1
        else:
            if row['result'] == 'A':
                home_wins += 1
            else:
                away_wins += 1

    return home_wins, draws, away_wins


def discretize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretiza features numéricas em categorias.

    Args:
        df: DataFrame com features numéricas

    Returns:
        DataFrame com features discretizadas adicionadas
    """
    # Forma recente (pontos)
    df['ht_form'] = pd.cut(
        df['ht_form_points'],
        bins=[-1, 3, 6, 9, 12, 15],
        labels=['poor', 'below avg', 'average', 'good', 'excellent']
    )
    df['at_form'] = pd.cut(
        df['at_form_points'],
        bins=[-1, 3, 6, 9, 12, 15],
        labels=['poor', 'below avg', 'average', 'good', 'excellent']
    )

    # Capacidade ofensiva (golos marcados)
    df['ht_attack'] = pd.cut(
        df['ht_form_goals_scored'],
        bins=[-1, 3, 7, 10, 100],
        labels=['weak', 'average', 'above avg', 'strong']
    )
    df['at_attack'] = pd.cut(
        df['at_form_goals_scored'],
        bins=[-1, 3, 7, 10, 100],
        labels=['weak', 'average', 'above avg', 'strong']
    )

    # Ataque conjunto
    df['joint_attack'] = df['ht_attack'].astype(str) + " x " + df['at_attack'].astype(str)

    # Dominância head-to-head
    df['h2h_dominance'] = df.apply(
        lambda r: 'home_dominant' if r['h2h_home_wins'] > r['h2h_away_wins'] + 3
        else ('away_dominant' if r['h2h_away_wins'] > r['h2h_home_wins'] + 3
              else 'balanced'),
        axis=1
    )

    # Favorito segundo odds
    df['odds_favorite'] = df.apply(
        lambda r: 'home' if r['h_odds'] < r['a_odds'] - 0.5
        else ('away' if r['a_odds'] < r['h_odds'] - 0.5 else 'balanced'),
        axis=1
    )

    return df


def prepare_dataset(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Pipeline completo de preparação do dataset.

    Args:
        input_path: Caminho para o CSV de entrada
        output_path: Caminho para salvar o dataset processado (opcional)

    Returns:
        DataFrame processado com todas as features
    """
    print("="*70)
    print("PART 0: Data Pipeline and Feature Engineering")
    print("="*70)

    # Carregar dados
    df = load_dataset(input_path)

    # Calcular features
    print("\nCalculando features de forma recente...")
    df = compute_form_features(df, n_matches=5)

    # Filtrar jogos após 2005
    df = df.loc[df['date'].dt.year > 2005, :].reset_index(drop=True)

    # Preencher NaN com 0
    df = df.fillna(0)

    # Discretizar features
    print("Discretizando features...")
    df = discretize_features(df)

    # Estatísticas finais
    print(f"\nDataset final: {len(df)} jogos")
    print(f"Features: {len(df.columns)} colunas")
    print(f"\nDistribuição de resultados:")
    print(df['result'].value_counts())

    # Salvar se solicitado
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nDataset salvo em: {output_path}")

    return df


def temporal_split(df: pd.DataFrame, split_year: int = 2022) -> tuple:
    """
    Divide o dataset em treino e teste por data.

    Args:
        df: DataFrame completo
        split_year: Ano de divisão (treino < split_year, teste >= split_year)

    Returns:
        Tupla (df_train, df_test)
    """
    train = df[df['date'].dt.year < split_year].reset_index(drop=True)
    test = df[df['date'].dt.year >= split_year].reset_index(drop=True)

    print(f"\nTemporal Split:")
    print(f"  Treino: {len(train)} jogos ({train['date'].min()} a {train['date'].max()})")
    print(f"  Teste:  {len(test)} jogos ({test['date'].min()} a {test['date'].max()})")

    return train, test


if __name__ == "__main__":
    # Exemplo de uso
    df = prepare_dataset(
        input_path="../data/dataset.csv",
        output_path="../data/dataset_processed.csv"
    )

    train, test = temporal_split(df, split_year=2022)
