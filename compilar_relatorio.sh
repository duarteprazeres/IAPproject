#!/bin/bash
# ============================================================
# Script para compilar o relatório LaTeX em PDF
# ============================================================

set -e

REPORT="relatorio"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# ---------- Verificar pdflatex ----------
PDFLATEX=""
for candidate in \
    pdflatex \
    /Library/TeX/texbin/pdflatex \
    /usr/local/texlive/2024/bin/universal-darwin/pdflatex \
    /usr/local/texlive/2023/bin/universal-darwin/pdflatex \
    /usr/local/texlive/2022/bin/x86_64-darwin/pdflatex \
    /usr/texbin/pdflatex; do
    if command -v "$candidate" &>/dev/null; then
        PDFLATEX="$candidate"
        break
    fi
done

if [ -z "$PDFLATEX" ]; then
    echo "❌  pdflatex não encontrado."
    echo ""
    echo "Por favor instale o MacTeX:"
    echo "   brew install --cask mactex"
    echo ""
    echo "Ou o BasicTeX (mais leve):"
    echo "   brew install --cask basictex"
    echo "   sudo tlmgr update --self"
    echo "   sudo tlmgr install collection-fontsrecommended babel-portuguese hyphen-portuguese"
    echo ""
    echo "Após a instalação, adicione ao PATH:"
    echo "   export PATH=\"\$PATH:/Library/TeX/texbin\""
    echo "E execute este script novamente."
    exit 1
fi

echo "✅  pdflatex encontrado: $PDFLATEX"
echo "📄  A compilar $REPORT.tex ..."

# Primeira passagem
"$PDFLATEX" -interaction=nonstopmode "$REPORT.tex" > /dev/null 2>&1 || true

# Segunda passagem (para referências cruzadas, índice)
"$PDFLATEX" -interaction=nonstopmode "$REPORT.tex" > /dev/null 2>&1 || true

# Terceira passagem (para garantir consistência)
"$PDFLATEX" -interaction=nonstopmode "$REPORT.tex"

if [ -f "$REPORT.pdf" ]; then
    echo ""
    echo "✅  PDF gerado com sucesso: $DIR/$REPORT.pdf"
    echo "📂  A abrir o PDF..."
    open "$REPORT.pdf" 2>/dev/null || echo "   (Abrir manualmente: $DIR/$REPORT.pdf)"
else
    echo "❌  Falhou a gerar o PDF. Verifique o ficheiro $REPORT.log para detalhes."
    exit 1
fi

# Limpeza de ficheiros auxiliares
echo "🧹  A limpar ficheiros auxiliares..."
rm -f "$REPORT.aux" "$REPORT.log" "$REPORT.out" "$REPORT.toc"
echo "✅  Concluído!"
