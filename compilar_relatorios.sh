#!/bin/bash
# Compila os relatórios LaTeX do projecto IAP.
#
# Uso:
#   ./compilar_relatorios.sh            # compila ambos
#   ./compilar_relatorios.sh notebook   # só versão notebook
#   ./compilar_relatorios.sh python     # só versão python

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- Localizar pdflatex ----------
PDFLATEX=""
for candidate in pdflatex \
    /Library/TeX/texbin/pdflatex \
    /usr/local/texlive/2024/bin/universal-darwin/pdflatex \
    /usr/local/texlive/2023/bin/universal-darwin/pdflatex; do
    if command -v "$candidate" &>/dev/null; then
        PDFLATEX="$candidate"
        break
    fi
done

if [ -z "$PDFLATEX" ]; then
    echo "pdflatex não encontrado. Instale o MacTeX:"
    echo "  brew install --cask mactex"
    echo "E adicione ao PATH: export PATH=\"\$PATH:/Library/TeX/texbin\""
    exit 1
fi

# ---------- Função de compilação ----------
compile() {
    local label="$1"   # "notebook" ou "python"
    local tex_dir="$DIR/$label/docs"
    local tex_file="relatorio_$label.tex"

    echo ""
    echo "=== A compilar: $label ($tex_dir/$tex_file) ==="

    if [ ! -f "$tex_dir/$tex_file" ]; then
        echo "Ficheiro não encontrado: $tex_dir/$tex_file"
        return 1
    fi

    cd "$tex_dir"

    # Duas passagens para resolver referências cruzadas e índice
    "$PDFLATEX" -interaction=nonstopmode "$tex_file" > /dev/null 2>&1 || true
    "$PDFLATEX" -interaction=nonstopmode "$tex_file"

    # Limpar ficheiros auxiliares
    rm -f "${tex_file%.tex}.aux" "${tex_file%.tex}.log" \
          "${tex_file%.tex}.out" "${tex_file%.tex}.toc"

    local pdf="${tex_file%.tex}.pdf"
    if [ -f "$pdf" ]; then
        echo "PDF gerado: $tex_dir/$pdf"
    else
        echo "Erro ao gerar PDF. Verifique o ficheiro .log."
        return 1
    fi

    cd "$DIR"
}

# ---------- Seleccionar o que compilar ----------
TARGET="${1:-both}"

case "$TARGET" in
    notebook) compile notebook ;;
    python)   compile python ;;
    both)     compile notebook; compile python ;;
    *)
        echo "Uso: $0 [notebook|python|both]"
        exit 1
        ;;
esac

echo ""
echo "Concluído."
