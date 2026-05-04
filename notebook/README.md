# Versão Jupyter Notebook - Projeto IAP

Este diretório contém a versão em Jupyter Notebook do projeto de Inteligência Artificial Probabilística.

## Estrutura

```
notebook/
├── Part0_Data_Pipeline.ipynb              # Pipeline de dados e análise exploratória
├── Part1_Bayesian_Networks.ipynb          # Redes Bayesianas (KE e Hill-Climbing)
├── Part2_Classification_Uncertainty.ipynb # Classificação probabilística e incerteza
├── Part3_Utility_Decision.ipynb           # Teoria de utilidade e tomada de decisão
├── data/
│   ├── dataset.csv                        # Dataset processado
│   └── requirements.txt                   # Dependências Python
├── plots/                                 # Visualizações geradas
└── docs/
    └── relatorio_notebook.tex             # Relatório LaTeX desta versão

```

## Como Executar

### 1. Instalação de Dependências

```bash
pip install -r data/requirements.txt
```

### 2. Execução dos Notebooks

Os notebooks devem ser executados **sequencialmente** na seguinte ordem:

1. **Part0_Data_Pipeline.ipynb** - Carrega e processa os dados, gera features
2. **Part1_Bayesian_Networks.ipynb** - Constrói e analisa Redes Bayesianas
3. **Part2_Classification_Uncertainty.ipynb** - Treina classificadores com quantificação de incerteza
4. **Part3_Utility_Decision.ipynb** - Avalia utilidade e estratégias de decisão

### 3. Compilação do Relatório

```bash
cd docs/
pdflatex relatorio_notebook.tex
pdflatex relatorio_notebook.tex  # Segunda execução para referências cruzadas
```

## Características da Versão Notebook

- **Interatividade**: Permite experimentação célula a célula
- **Visualizações inline**: Gráficos e tabelas embutidos no notebook
- **Documentação integrada**: Texto markdown com código e resultados
- **Exploração facilitada**: Ideal para análise exploratória e apresentações

## Dependências Principais

- Python 3.8+
- pandas, numpy
- scikit-learn
- pgmpy (Redes Bayesianas)
- MAPIE (Conformal Prediction)
- matplotlib, seaborn
- scikit-optimize (Otimização Bayesiana)

## Resultados Principais

- **Parte 0**: Dataset com 5,583 jogos e 21 features
- **Parte 1**: Acurácia ~51% com Redes Bayesianas
- **Parte 2**: Melhor acurácia 56.95% (Logistic Regression otimizada)
- **Parte 3**: Estratégia híbrida atinge 58.07% de acurácia

## Autor

Duarte Prazeres (2023214962)
Inteligência Artificial Probabilística 2025/2026
