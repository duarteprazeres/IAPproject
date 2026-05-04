# Versão Python Modular - Projeto IAP

Este diretório contém a versão modular em Python do projeto de Inteligência Artificial Probabilística.

## Estrutura do Projeto

```
python/
├── main.py                          # Script principal que orquestra todo o pipeline
├── src/
│   ├── __init__.py                  # Inicialização do pacote
│   ├── data_pipeline.py             # Part 0: Pipeline de dados e feature engineering
│   ├── bayesian_networks.py         # Part 1: Redes Bayesianas (KE e Hill-Climbing)
│   ├── classification.py            # Part 2: Classificação probabilística e incerteza
│   └── utility_decision.py          # Part 3: Teoria de utilidade e tomada de decisão
├── data/
│   ├── dataset.csv                  # Dataset original
│   ├── dataset_processed.csv        # Dataset processado (gerado)
│   ├── best_model_proba_test.npy   # Probabilidades do melhor modelo (gerado)
│   ├── best_model_classes.npy      # Classes do modelo (gerado)
│   └── requirements.txt             # Dependências Python
├── plots/                           # Visualizações (se implementadas)
└── docs/
    └── relatorio_python.tex         # Relatório LaTeX desta versão
```

## Características da Versão Python

- **Modularidade**: Código organizado em módulos reutilizáveis
- **Orientação a Objetos**: Classes para encapsular funcionalidades
- **Command-Line Interface**: Script principal com argumentos flexíveis
- **Separação de Responsabilidades**: Cada módulo tem uma função clara
- **Testabilidade**: Estrutura facilita testes unitários
- **Produção-Ready**: Adequado para deploy em ambientes de produção

## Instalação

### 1. Criar Ambiente Virtual (Recomendado)

```bash
cd python
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 2. Instalar Dependências

```bash
pip install -r data/requirements.txt
```

## Como Executar

### Opção 1: Pipeline Completo

Executa todas as 4 partes sequencialmente:

```bash
python main.py
```

### Opção 2: Executar Apenas Uma Parte

```bash
# Apenas Part 0 (Data Pipeline)
python main.py --only-part 0

# Apenas Part 1 (Bayesian Networks)
python main.py --only-part 1

# Apenas Part 2 (Classification)
python main.py --only-part 2

# Apenas Part 3 (Utility & Decision)
python main.py --only-part 3
```

### Opção 3: Pular Uma Parte

```bash
# Executar todas exceto Part 1
python main.py --skip-part 1
```

### Opção 4: Usar Módulos Individualmente

```python
# Importar e usar módulos diretamente
from src.data_pipeline import prepare_dataset, temporal_split
from src.bayesian_networks import run_bayesian_networks_analysis

# Preparar dados
df = prepare_dataset('data/dataset.csv')
train, test = temporal_split(df, split_year=2022)

# Executar análise Bayesiana
results = run_bayesian_networks_analysis(train, test)
```

## Descrição dos Módulos

### `src/data_pipeline.py` (Part 0)

Funções principais:
- `load_dataset()`: Carrega dataset do CSV
- `compute_form_features()`: Calcula features de forma recente (últimos 5 jogos)
- `discretize_features()`: Discretiza variáveis numéricas
- `prepare_dataset()`: Pipeline completo de preparação
- `temporal_split()`: Divisão temporal treino/teste

**Saída**: Dataset com 5,583 jogos e 21 features

### `src/bayesian_networks.py` (Part 1)

Classe principal: `FootballBayesianNetwork`

Métodos:
- `build_knowledge_engineering_structure()`: Estrutura manual baseada em domínio
- `learn_structure_hill_climbing()`: Aprendizagem automática com BIC score
- `fit_parameters()`: Estimação de parâmetros (MLE ou Bayesian)
- `analyze_d_separation()`: Análise de independências condicionais
- `perform_inference()`: Inferência probabilística com Variable Elimination
- `evaluate_on_test()`: Avaliação no conjunto de teste

**Saída**: Modelos KE e HC com acurácia ~51%

### `src/classification.py` (Part 2)

Classe principal: `ProbabilisticClassifier`

Métodos:
- `train_gaussian_nb()`: Naive Bayes Gaussiano
- `train_categorical_nb()`: Naive Bayes Categórico
- `train_logistic_regression()`: Regressão Logística
- `conformal_prediction()`: Conformal Prediction para garantias de cobertura
- `calibrate_model()`: Calibração isotónica/sigmóide
- `bayesian_optimization()`: Otimização Bayesiana de hiperparâmetros

**Saída**: Melhor modelo com acurácia ~57%, probabilidades calibradas

### `src/utility_decision.py` (Part 3)

Classe principal: `UtilityAnalyzer`

Métodos:
- `extract_odds_probabilities()`: Converte odds em probabilidades implícitas
- `calculate_disagreement()`: Mede discordância modelo vs. mercado
- `baseline_odds_accuracy()`: Acurácia das odds (baseline)
- `reject_option_strategy()`: Estratégia de abstinência em casos incertos
- `betting_simulation()`: Simulação de value betting

**Saída**: Estratégia híbrida com acurácia ~58%, análise de P&L

## Fluxo de Dados

```
dataset.csv
    ↓
[Part 0: Data Pipeline]
    ↓
train_data, test_data
    ↓
    ├─→ [Part 1: Bayesian Networks] → modelos KE & HC
    │
    ├─→ [Part 2: Classification] → best_model_proba_test.npy
    │                               best_model_classes.npy
    ↓
[Part 3: Utility & Decision] → análise de utilidade, betting simulation
```

## Arquitetura de Classes

```
FootballBayesianNetwork
├── build_knowledge_engineering_structure()
├── learn_structure_hill_climbing()
├── fit_parameters()
├── analyze_d_separation()
├── perform_inference()
└── evaluate_on_test()

ProbabilisticClassifier
├── prepare_features()
├── train_gaussian_nb()
├── train_categorical_nb()
├── train_logistic_regression()
├── conformal_prediction()
├── calibrate_model()
├── bayesian_optimization()
└── evaluate()

UtilityAnalyzer
├── extract_odds_probabilities()
├── calculate_disagreement()
├── baseline_odds_accuracy()
├── reject_option_strategy()
└── betting_simulation()
```

## Dependências Principais

```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
pgmpy >= 0.1.19          # Redes Bayesianas
MAPIE >= 0.6.0           # Conformal Prediction
scikit-optimize >= 0.9.0 # Otimização Bayesiana
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

## Resultados Principais

| Parte | Métrica | Resultado |
|-------|---------|-----------|
| Part 0 | Dataset | 5,583 jogos, 21 features |
| Part 1 | Acurácia RB | ~51% (KE e HC) |
| Part 2 | Melhor Acurácia | 56.95% (LR otimizada) |
| Part 2 | Log Loss | 0.9315 |
| Part 2 | Conformal Coverage | 95.3% (@95% conf.) |
| Part 3 | Odds Baseline | 58.0% |
| Part 3 | Estratégia Híbrida | 58.07% |
| Part 3 | Reject Option | 60.2% (@60% coverage) |

## Compilação do Relatório

```bash
cd docs/
pdflatex relatorio_python.tex
pdflatex relatorio_python.tex  # Segunda execução para referências
```

## Vantagens vs. Versão Notebook

1. **Performance**: Execução mais rápida sem overhead do Jupyter
2. **Automação**: Fácil integração em pipelines de CI/CD
3. **Escalabilidade**: Estrutura modular facilita expansão
4. **Manutenibilidade**: Código mais organizado e testável
5. **Deploy**: Adequado para produção (API, batch jobs, etc.)
6. **Version Control**: Melhor integração com Git

## Extensões Possíveis

1. **Visualizações**: Adicionar módulo `visualization.py` para gráficos
2. **Logging**: Implementar logging estruturado
3. **Testes**: Adicionar `tests/` com pytest
4. **API REST**: Expor modelos via Flask/FastAPI
5. **Config Files**: Usar YAML para configuração
6. **Async**: Processar jogos em paralelo com asyncio
7. **Database**: Integrar com PostgreSQL/MongoDB
8. **MLOps**: Integrar MLflow para tracking de experimentos

## Autor

**Duarte Prazeres** (2023214962)
Inteligência Artificial Probabilística 2025/2026
Universidade de Coimbra

## Licença

Este projeto é parte do trabalho académico da UC de Inteligência Artificial Probabilística.
