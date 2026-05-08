# Projeto IAP 2025/2026 - Previsão Probabilística de Futebol

**Autor:** Duarte Prazeres (2023214962)
**Disciplina:** Inteligência Artificial Probabilística


## Visão Geral

Este repositório contém o projeto final da disciplina de Inteligência Artificial Probabilística, focado no desenvolvimento de um sistema de previsão probabilística para resultados de jogos da Liga Portuguesa de Futebol.

O projeto implementa técnicas de:
- **Redes Bayesianas** (Knowledge Engineering e Hill-Climbing)
- **Classificação Probabilística** (Naive Bayes, Logistic Regression)
- **Quantificação de Incerteza** (Conformal Prediction, Calibração)
- **Teoria de Utilidade** (Análise de odds, Reject Option, Value Betting)

## Estrutura do Repositório

Este repositório está organizado em **duas versões completas** do projeto:

```
IAPproject/
│
├── compilar_relatorios.sh           # Script de compilação dos relatórios LaTeX
├── dataset/                         # Dataset único (partilhado por ambas as versões)
│   └── dataset.csv
│
├── notebook/                        # 📓 VERSÃO JUPYTER NOTEBOOK
│   ├── Part0_Data_Pipeline.ipynb
│   ├── Part1_Bayesian_Networks.ipynb
│   ├── Part2_Classification_Uncertainty.ipynb
│   ├── Part3_Utility_Decision.ipynb
│   ├── data/
│   │   └── requirements.txt
│   ├── plots/
│   └── docs/
│       └── relatorio_notebook.tex   # Relatório LaTeX da versão notebook
│
├── python/                          # 🐍 VERSÃO PYTHON MODULAR
│   ├── main.py                      # Script principal CLI
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_pipeline.py
│   │   ├── bayesian_networks.py
│   │   ├── classification.py
│   │   └── utility_decision.py
│   ├── data/
│   │   └── requirements.txt
│   ├── plots/
│   └── docs/
│       └── relatorio_python.tex     # Relatório LaTeX da versão Python
│
└── README.md                        # Este ficheiro
```

## Duas Versões, Mesma Análise

Ambas as versões implementam **exatamente a mesma análise técnica** com os mesmos algoritmos e resultados. A diferença está na **organização do código**:

### Versão Jupyter Notebook

**Localização:** `notebook/`

**Características:**
- Código organizado em 4 notebooks sequenciais
- Visualizações inline integradas
- Documentação Markdown intercalada
- Narrativa linear e pedagógica
- Ideal para exploração e apresentações
- Execução célula a célula

**Ideal para:**
- Análise exploratória de dados
- Demonstrações e apresentações
- Ensino e aprendizagem
- Prototipagem rápida
- Documentação integrada

**Como executar:**

macOS / Linux:
```bash
cd notebook/
python3 -m venv venv && source venv/bin/activate
pip install -r data/requirements.txt
jupyter notebook
# Executar sequencialmente: Part0 → Part1 → Part2 → Part3
```

Windows:
```cmd
cd notebook/
python -m venv venv && venv\Scripts\activate
pip install -r data/requirements.txt
jupyter notebook
```

**Relatório:** `notebook/docs/relatorio_notebook.tex`

---

### Versão Python Modular

**Localização:** `python/`

**Características:**
- Arquitetura modular com classes bem definidas
- Separação de responsabilidades (data, models, analysis)
- Interface CLI com argumentos flexíveis
- Código reutilizável e testável
- Production-ready
- Melhor integração com Git e CI/CD

**Ideal para:**
- Deploy em produção
- Integração com sistemas existentes
- Testes automatizados
- Pipelines de ML
- Versionamento robusto
- Escalabilidade

**Como executar:**

macOS / Linux:
```bash
cd python/
python3 -m venv venv && source venv/bin/activate
pip install -r data/requirements.txt
python main.py                  # pipeline completo
python main.py --only-part 2    # apenas uma parte
python main.py --skip-part 1    # pular uma parte
```

Windows:
```cmd
cd python/
python -m venv venv && venv\Scripts\activate
pip install -r data/requirements.txt
python main.py
```

**Relatório:** `python/docs/relatorio_python.tex`

---

## Conteúdo Técnico

Ambas as versões implementam as 4 partes do projeto:

### Parte 0: Pipeline de Dados e EDA
- Carregamento e limpeza de dados
- Feature engineering (forma recente, H2H, odds)
- Discretização de variáveis
- Análise exploratória

### Parte 1: Redes Bayesianas
- Knowledge Engineering (estrutura manual)
- Hill-Climbing (aprendizagem automática)
- D-separação e independências condicionais
- Estimação de parâmetros (MLE vs. Bayesian)
- Inferência probabilística

### Parte 2: Classificação e Incerteza
- Naive Bayes (Gaussiano e Categórico)
- Logistic Regression
- Conformal Prediction (garantias de cobertura)
- Calibração (isotónica e sigmóide)
- Otimização Bayesiana de hiperparâmetros

### Parte 3: Utilidade e Decisão
- Probabilidades implícitas das odds
- Análise de discordância modelo vs. mercado
- Reject Option (estratégia de abstinência)
- Estratégia híbrida (modelo + odds)
- Simulação de apostas (value betting)

## Resultados Principais

| Métrica | Valor |
|---------|-------|
| Dataset Total | 5,583 jogos (2006-2026) |
| Features | 21 (15 numéricas + 6 categóricas) |
| Treino | 4,331 jogos (2006-2021) |
| Teste | 1,252 jogos (2022-2026) |
| **Rede Bayesiana** | 51% acurácia |
| **Naive Bayes Categórico** | 56.8% acurácia |
| **Logistic Regression Otimizada** | 56.95% acurácia |
| **Conformal Prediction** | 95.3% cobertura (@95% conf.) |
| **Odds do Mercado (Baseline)** | 58.0% acurácia |
| **Estratégia Híbrida** | **58.07% acurácia** |
| **Reject Option** | 60.2% (@60% cobertura) |

## Dependências

Ambas as versões partilham as mesmas dependências Python.

Instalar (usando a versão desejada):
```bash
pip install -r notebook/data/requirements.txt   # versão notebook
pip install -r python/data/requirements.txt     # versão python
```

## Compilação dos Relatórios

Usar o script fornecido (recomendado):
```bash
./compilar_relatorios.sh             # compila ambos
./compilar_relatorios.sh notebook    # só versão notebook
./compilar_relatorios.sh python      # só versão python
```

Ou manualmente (duas passagens para resolver referências cruzadas):
```bash
cd notebook/docs/ && pdflatex relatorio_notebook.tex && pdflatex relatorio_notebook.tex
cd python/docs/   && pdflatex relatorio_python.tex   && pdflatex relatorio_python.tex
```

## Dataset

O dataset contém resultados de jogos da Liga Portuguesa de Futebol entre 2006 e 2026, incluindo:
- Resultados (golos casa/fora, resultado final)
- Odds de apostas (casa, empate, fora)
- Features de forma recente (últimos 5 jogos)
- Estatísticas head-to-head
- Variáveis discretizadas

**Fonte:** [football-data.co.uk](https://www.football-data.co.uk/portugalm.php)

## Ficheiros Principais

### Raiz do Repositório
- `dataset/dataset.csv` - Dataset completo (partilhado por ambas as versões)
- `README.md` - Este ficheiro

### Versão Notebook (`notebook/`)
- `Part0_Data_Pipeline.ipynb` - Pipeline e EDA
- `Part1_Bayesian_Networks.ipynb` - Redes Bayesianas
- `Part2_Classification_Uncertainty.ipynb` - Classificação
- `Part3_Utility_Decision.ipynb` - Utilidade
- `data/requirements.txt` - Dependências Python
- `plots/` - Visualizações geradas
- `docs/relatorio_notebook.tex` - Relatório LaTeX

### Versão Python (`python/`)
- `main.py` - Script principal CLI
- `src/data_pipeline.py` - Módulo de dados
- `src/bayesian_networks.py` - Módulo de RB
- `src/classification.py` - Módulo de classificação
- `src/utility_decision.py` - Módulo de utilidade
- `data/requirements.txt` - Dependências Python
- `plots/` - Visualizações geradas
- `docs/relatorio_python.tex` - Relatório LaTeX

## Contacto
Nome: Duarte Prazeres
E-mail: duartepcm@gmail.com
Número de Estudante: 2023214962

