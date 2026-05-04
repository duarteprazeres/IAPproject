# Projeto IAP 2025/2026 - Previsão Probabilística de Futebol

**Autor:** Duarte Prazeres (2023214962)
**Disciplina:** Inteligência Artificial Probabilística
**Universidade de Coimbra**

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
├── notebook/                        # 📓 VERSÃO JUPYTER NOTEBOOK
│   ├── Part0_Data_Pipeline.ipynb
│   ├── Part1_Bayesian_Networks.ipynb
│   ├── Part2_Classification_Uncertainty.ipynb
│   ├── Part3_Utility_Decision.ipynb
│   ├── data/
│   │   ├── dataset.csv
│   │   └── requirements.txt
│   ├── plots/
│   └── docs/
│       └── relatorio_notebook.tex   # Relatório LaTeX da versão notebook
│
├── python/                          # 🐍 VERSÃO PYTHON MODULAR
│   ├── main.py                      # Script principal CLI
│   ├── src/
│   │   ├── data_pipeline.py
│   │   ├── bayesian_networks.py
│   │   ├── classification.py
│   │   └── utility_decision.py
│   ├── data/
│   │   ├── dataset.csv
│   │   └── requirements.txt
│   ├── plots/
│   └── docs/
│       └── relatorio_python.tex     # Relatório LaTeX da versão Python
│
├── dataset.csv                      # Dataset original (raiz)
├── relatorio.tex                    # Relatório original
├── relatorio.pdf                    # PDF compilado
└── README.md                        # Este ficheiro
```

## Duas Versões, Mesma Análise

Ambas as versões implementam **exatamente a mesma análise técnica** com os mesmos algoritmos e resultados. A diferença está na **organização do código**:

### 📓 Versão Jupyter Notebook

**Localização:** `notebook/`

**Características:**
- ✅ Código organizado em 4 notebooks sequenciais
- ✅ Visualizações inline integradas
- ✅ Documentação Markdown intercalada
- ✅ Narrativa linear e pedagógica
- ✅ Ideal para exploração e apresentações
- ✅ Execução célula a célula

**Quando usar:**
- Análise exploratória de dados
- Demonstrações e apresentações
- Ensino e aprendizagem
- Prototipagem rápida
- Documentação integrada

**Como executar:**
```bash
cd notebook/
pip install -r data/requirements.txt
jupyter notebook
# Executar sequencialmente: Part0 → Part1 → Part2 → Part3
```

**Relatório:** `notebook/docs/relatorio_notebook.tex`

---

### 🐍 Versão Python Modular

**Localização:** `python/`

**Características:**
- ✅ Arquitetura modular com classes bem definidas
- ✅ Separação de responsabilidades (data, models, analysis)
- ✅ Interface CLI com argumentos flexíveis
- ✅ Código reutilizável e testável
- ✅ Production-ready
- ✅ Melhor integração com Git e CI/CD

**Quando usar:**
- Deploy em produção
- Integração com sistemas existentes
- Testes automatizados
- Pipelines de ML
- Versionamento robusto
- Escalabilidade

**Como executar:**
```bash
cd python/
pip install -r data/requirements.txt

# Pipeline completo
python main.py

# Apenas uma parte específica
python main.py --only-part 2

# Pular uma parte
python main.py --skip-part 1
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
- **Resultado:** 5,583 jogos, 21 features

### Parte 1: Redes Bayesianas
- Knowledge Engineering (estrutura manual)
- Hill-Climbing (aprendizagem automática)
- D-separação e independências condicionais
- Estimação de parâmetros (MLE vs. Bayesian)
- Inferência probabilística
- **Resultado:** Acurácia ~51%

### Parte 2: Classificação e Incerteza
- Naive Bayes (Gaussiano e Categórico)
- Logistic Regression
- Conformal Prediction (garantias de cobertura)
- Calibração (isotónica e sigmóide)
- Otimização Bayesiana de hiperparâmetros
- **Resultado:** Acurácia 56.95% (LR otimizada)

### Parte 3: Utilidade e Decisão
- Probabilidades implícitas das odds
- Análise de discordância modelo vs. mercado
- Reject Option (estratégia de abstinência)
- Estratégia híbrida (modelo + odds)
- Simulação de apostas (value betting)
- **Resultado:** Estratégia híbrida 58.07%

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

Ambas as versões partilham as mesmas dependências Python:

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

Instalar em qualquer versão:
```bash
pip install -r [notebook|python]/data/requirements.txt
```

## Compilação dos Relatórios

### Versão Notebook:
```bash
cd notebook/docs/
pdflatex relatorio_notebook.tex
pdflatex relatorio_notebook.tex  # Segunda execução
```

### Versão Python:
```bash
cd python/docs/
pdflatex relatorio_python.tex
pdflatex relatorio_python.tex
```

### Relatório Original (raiz):
```bash
pdflatex relatorio.tex
pdflatex relatorio.tex
```

Ou usar o script fornecido:
```bash
./compilar_relatorio.sh
```

## Comparação: Notebook vs. Python

| Aspecto | Jupyter Notebook | Python Modular |
|---------|-----------------|----------------|
| **Organização** | 4 notebooks sequenciais | 4 módulos + main.py |
| **Execução** | Célula a célula | CLI com argumentos |
| **Visualizações** | Inline | Salvas em arquivos |
| **Documentação** | Markdown integrado | Docstrings + README |
| **Versionamento** | JSON (difícil diff) | Python puro (fácil diff) |
| **Testes** | Manual | Automatizável (pytest) |
| **Reutilização** | Copy-paste | Import de classes |
| **Produção** | Não recomendado | Adequado |
| **Exploração** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Manutenção** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Apresentação** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## Escolhendo a Versão Certa

**Use a versão Notebook se:**
- Está a explorar os dados pela primeira vez
- Precisa criar uma apresentação ou demo
- Quer entender o código passo a passo
- Prefere ver visualizações inline
- Está a ensinar ou aprender

**Use a versão Python se:**
- Vai colocar em produção
- Precisa integrar com outros sistemas
- Quer escrever testes automatizados
- Trabalha em equipa com Git
- Precisa de performance e escalabilidade
- Quer reutilizar componentes noutros projetos

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
- `dataset.csv` - Dataset original completo
- `relatorio.tex` - Relatório LaTeX original
- `relatorio.pdf` - PDF compilado do relatório
- `compilar_relatorio.sh` - Script de compilação LaTeX
- `requirements.txt` - Dependências Python
- `1_load_and_clean.py` - Script auxiliar de carregamento
- `2_feature_engineering.py` - Script auxiliar de features
- `plots/` - Visualizações originais

### Versão Notebook (`notebook/`)
- `Part0_Data_Pipeline.ipynb` - Pipeline e EDA
- `Part1_Bayesian_Networks.ipynb` - Redes Bayesianas
- `Part2_Classification_Uncertainty.ipynb` - Classificação
- `Part3_Utility_Decision.ipynb` - Utilidade
- `docs/relatorio_notebook.tex` - Relatório LaTeX

### Versão Python (`python/`)
- `main.py` - Script principal CLI
- `src/data_pipeline.py` - Módulo de dados
- `src/bayesian_networks.py` - Módulo de RB
- `src/classification.py` - Módulo de classificação
- `src/utility_decision.py` - Módulo de utilidade
- `docs/relatorio_python.tex` - Relatório LaTeX

## Fluxo de Trabalho Recomendado

1. **Exploração Inicial:** Use a versão **Notebook**
   ```bash
   cd notebook/
   jupyter notebook
   ```

2. **Produção/Deploy:** Use a versão **Python**
   ```bash
   cd python/
   python main.py
   ```

3. **Relatório:** Compile a versão apropriada
   ```bash
   cd [notebook|python]/docs/
   pdflatex relatorio_[notebook|python].tex
   ```

## Contacto

**Duarte Prazeres**
Número: 2023214962
Ano Letivo: 2025/2026

## Licença

Este projeto é parte do trabalho académico da UC de Inteligência Artificial Probabilística e está sujeito às regras de propriedade intelectual da Universidade de Coimbra.

---

**Nota:** Ambas as versões produzem resultados idênticos. A escolha entre elas depende do contexto de uso (exploração vs. produção).