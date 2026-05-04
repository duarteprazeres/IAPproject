"""
Part 1: Bayesian Networks

Este módulo implementa a construção, aprendizagem e inferência em Redes Bayesianas
para previsão de resultados de futebol.
"""

import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination


class FootballBayesianNetwork:
    """
    Classe para construção e análise de Redes Bayesianas para futebol.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa a rede com os dados.

        Args:
            data: DataFrame com variáveis discretas
        """
        self.data = data
        self.model_ke = None
        self.model_hc = None
        self.inference_ke = None
        self.inference_hc = None

    def build_knowledge_engineering_structure(self) -> BayesianNetwork:
        """
        Constrói estrutura da rede usando Knowledge Engineering (manual).

        Returns:
            Modelo de Rede Bayesiana com estrutura KE
        """
        print("\n" + "="*70)
        print("Construindo estrutura por Knowledge Engineering...")
        print("="*70)

        # Definir arestas baseadas em conhecimento do domínio
        edges = [
            # Forma influencia resultado
            ('ht_form', 'result'),
            ('at_form', 'result'),

            # Capacidade ofensiva influencia resultado
            ('ht_attack', 'result'),
            ('at_attack', 'result'),

            # Head-to-head influencia resultado
            ('h2h_dominance', 'result'),

            # Odds refletem expectativa de resultado
            ('odds_favorite', 'result'),

            # Forma influencia capacidade ofensiva
            ('ht_form', 'ht_attack'),
            ('at_form', 'at_attack'),
        ]

        self.model_ke = BayesianNetwork(edges)
        print(f"Estrutura KE criada com {len(edges)} arestas")
        print(f"Nós: {self.model_ke.nodes()}")

        return self.model_ke

    def learn_structure_hill_climbing(self, scoring_method='bic', max_indegree=3) -> BayesianNetwork:
        """
        Aprende estrutura da rede usando Hill-Climbing.

        Args:
            scoring_method: Critério de pontuação ('bic', 'k2', 'bdeu')
            max_indegree: Número máximo de pais por nó

        Returns:
            Modelo de Rede Bayesiana com estrutura aprendida
        """
        print("\n" + "="*70)
        print("Aprendendo estrutura por Hill-Climbing...")
        print("="*70)

        # Selecionar variáveis discretas relevantes
        discrete_vars = ['ht_form', 'at_form', 'ht_attack', 'at_attack',
                        'h2h_dominance', 'odds_favorite', 'result']

        data_discrete = self.data[discrete_vars].copy()

        # Converter para strings (pgmpy exige)
        for col in discrete_vars:
            data_discrete[col] = data_discrete[col].astype(str)

        # Configurar algoritmo de busca
        if scoring_method == 'bic':
            scoring = BicScore(data_discrete)
        else:
            raise ValueError(f"Scoring method '{scoring_method}' não implementado")

        hc = HillClimbSearch(data_discrete)

        # Executar busca
        print(f"Executando Hill-Climbing com {scoring_method.upper()} score...")
        best_model = hc.estimate(scoring_method=scoring, max_indegree=max_indegree)

        self.model_hc = best_model
        print(f"Estrutura aprendida com {len(best_model.edges())} arestas")
        print(f"Arestas: {best_model.edges()}")

        return self.model_hc

    def fit_parameters(self, model: BayesianNetwork, method='mle', **kwargs) -> BayesianNetwork:
        """
        Estima parâmetros (CPTs) da rede.

        Args:
            model: Modelo de Rede Bayesiana
            method: 'mle' (Maximum Likelihood) ou 'bayes' (Bayesian)
            **kwargs: Argumentos adicionais para o estimador

        Returns:
            Modelo com parâmetros ajustados
        """
        print(f"\nEstimando parâmetros usando {method.upper()}...")

        # Preparar dados discretos
        discrete_vars = list(model.nodes())
        data_discrete = self.data[discrete_vars].copy()

        for col in discrete_vars:
            data_discrete[col] = data_discrete[col].astype(str)

        # Escolher estimador
        if method == 'mle':
            estimator = MaximumLikelihoodEstimator(model, data_discrete)
            model.fit(data_discrete, estimator=estimator)

        elif method == 'bayes':
            prior_type = kwargs.get('prior_type', 'BDeu')
            equivalent_sample_size = kwargs.get('equivalent_sample_size', 10)

            estimator = BayesianEstimator(model, data_discrete)
            model.fit(data_discrete, estimator=estimator,
                     prior_type=prior_type,
                     equivalent_sample_size=equivalent_sample_size)

        else:
            raise ValueError(f"Method '{method}' não suportado")

        print("Parâmetros estimados com sucesso")
        return model

    def analyze_d_separation(self, model: BayesianNetwork) -> dict:
        """
        Analisa independências usando D-separação.

        Args:
            model: Modelo de Rede Bayesiana

        Returns:
            Dicionário com análises de independência
        """
        print("\n" + "="*70)
        print("Análise de D-Separação")
        print("="*70)

        independencies = {}

        # Independências locais para cada nó
        for node in model.nodes():
            local_indep = model.local_independencies(node)
            independencies[node] = str(local_indep)
            print(f"\n{node}: {local_indep}")

        # Markov Blanket
        print("\n" + "-"*70)
        print("Markov Blankets:")
        print("-"*70)

        for node in model.nodes():
            # Markov Blanket = pais + filhos + pais dos filhos
            parents = set(model.get_parents(node))
            children = set(model.get_children(node))
            children_parents = set()
            for child in children:
                children_parents.update(model.get_parents(child))

            markov_blanket = parents | children | (children_parents - {node})
            print(f"{node}: {markov_blanket}")

        return independencies

    def perform_inference(self, model: BayesianNetwork, queries: list) -> dict:
        """
        Realiza inferências probabilísticas na rede.

        Args:
            model: Modelo de Rede Bayesiana
            queries: Lista de dicionários com queries
                    Formato: {'evidence': {...}, 'description': '...'}

        Returns:
            Dicionário com resultados das queries
        """
        print("\n" + "="*70)
        print("Inferência Probabilística")
        print("="*70)

        # Inicializar inferência
        inference = VariableElimination(model)
        results = {}

        for i, query in enumerate(queries, 1):
            evidence = query.get('evidence', {})
            description = query.get('description', f'Query {i}')

            # Converter valores para strings
            evidence_str = {k: str(v) for k, v in evidence.items()}

            print(f"\n{description}")
            print(f"  Evidência: {evidence}")

            try:
                # Inferir P(result | evidence)
                result_dist = inference.query(variables=['result'], evidence=evidence_str)
                probs = result_dist.values

                # Extrair probabilidades
                states = result_dist.state_names['result']
                prob_dict = {state: float(prob) for state, prob in zip(states, probs)}

                print(f"  P(H) = {prob_dict.get('H', 0.0):.3f}")
                print(f"  P(D) = {prob_dict.get('D', 0.0):.3f}")
                print(f"  P(A) = {prob_dict.get('A', 0.0):.3f}")

                results[description] = prob_dict

            except Exception as e:
                print(f"  Erro na inferência: {e}")
                results[description] = None

        return results

    def evaluate_on_test(self, model: BayesianNetwork, test_data: pd.DataFrame) -> dict:
        """
        Avalia o modelo no conjunto de teste.

        Args:
            model: Modelo de Rede Bayesiana treinado
            test_data: Dados de teste

        Returns:
            Dicionário com métricas de avaliação
        """
        print("\n" + "="*70)
        print("Avaliação no Conjunto de Teste")
        print("="*70)

        inference = VariableElimination(model)
        predictions = []
        true_labels = []

        discrete_vars = list(model.nodes())
        test_discrete = test_data[discrete_vars].copy()

        for col in discrete_vars:
            test_discrete[col] = test_discrete[col].astype(str)

        for idx, row in test_discrete.iterrows():
            # Preparar evidência (todas as features exceto resultado)
            evidence = {k: v for k, v in row.items() if k != 'result'}

            try:
                # Inferir P(result | evidência)
                result_dist = inference.query(variables=['result'], evidence=evidence)
                pred = result_dist.state_names['result'][np.argmax(result_dist.values)]
                predictions.append(pred)
                true_labels.append(row['result'])

            except Exception:
                # Em caso de erro, prever a classe mais frequente
                predictions.append('H')
                true_labels.append(row['result'])

        # Calcular acurácia
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))

        print(f"Acurácia: {accuracy:.4f}")

        # Precisão/Recall por classe
        from sklearn.metrics import classification_report
        print("\n" + classification_report(true_labels, predictions))

        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels
        }


def run_bayesian_networks_analysis(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Executa análise completa de Redes Bayesianas.

    Args:
        train_data: Dados de treino
        test_data: Dados de teste
    """
    print("\n" + "="*70)
    print("PART 1: Bayesian Networks Analysis")
    print("="*70)

    # Inicializar
    bn = FootballBayesianNetwork(train_data)

    # 1. Knowledge Engineering
    model_ke = bn.build_knowledge_engineering_structure()
    model_ke = bn.fit_parameters(model_ke, method='mle')

    # 2. Hill-Climbing
    model_hc = bn.learn_structure_hill_climbing(scoring_method='bic', max_indegree=3)
    model_hc = bn.fit_parameters(model_hc, method='bayes', equivalent_sample_size=10)

    # 3. D-Separação (apenas KE para brevidade)
    bn.analyze_d_separation(model_ke)

    # 4. Queries de Inferência
    queries = [
        {
            'evidence': {'ht_form': 'good', 'odds_favorite': 'home'},
            'description': 'Q1: Equipa casa com boa forma e favorita nas odds'
        },
        {
            'evidence': {'ht_form': 'average', 'at_form': 'average'},
            'description': 'Q2: Ambas equipas com forma média'
        },
        {
            'evidence': {'h2h_dominance': 'home_dominant'},
            'description': 'Q3: Dominância H2H da equipa da casa'
        },
    ]

    print("\n" + "="*70)
    print("Inferências no Modelo KE")
    print("="*70)
    results_ke = bn.perform_inference(model_ke, queries)

    # 5. Avaliação
    print("\n\nAvaliação Modelo KE:")
    metrics_ke = bn.evaluate_on_test(model_ke, test_data)

    print("\n\nAvaliação Modelo HC:")
    metrics_hc = bn.evaluate_on_test(model_hc, test_data)

    return {
        'model_ke': model_ke,
        'model_hc': model_hc,
        'results_ke': results_ke,
        'metrics_ke': metrics_ke,
        'metrics_hc': metrics_hc
    }


if __name__ == "__main__":
    # Exemplo de uso
    from data_pipeline import load_dataset, temporal_split

    df = load_dataset("../data/dataset.csv")
    train, test = temporal_split(df, split_year=2022)

    results = run_bayesian_networks_analysis(train, test)
