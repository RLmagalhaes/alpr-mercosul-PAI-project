"""Converte RELATORIO.md em HTML, formatado para impressao em PDF.

Gera DOIS arquivos a partir do mesmo Markdown:

    RELATORIO.html          versao completa — a do repositorio/GitHub
    RELATORIO_entrega.html  sem as secoes listadas em SECOES_FORA_DA_ENTREGA
                            — esta e a que vai para a pasta `entrega/`

Assim o Markdown continua sendo a unica fonte da verdade: nada e editado
a mao em dois lugares.

Nao ha pandoc nesta maquina, entao a conversao usa o modulo `markdown` do
Python e um CSS proprio pensado para papel A4: margens, quebras de pagina antes
de cada secao, tabelas com cabecalho repetido e tipografia serifada.

Para gerar o PDF: abra o HTML no navegador e use Cmd+P -> "Salvar como PDF",
com "Imprimir planos de fundo" LIGADO (senao as tabelas perdem o fundo cinza).

Uso:
    .venv/bin/python notebooks/10_gerar_html_relatorio.py
"""

import os
import re

import markdown

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = f"{RAIZ}/RELATORIO.md"
SAIDA = f"{RAIZ}/RELATORIO.html"
SAIDA_ENTREGA = f"{RAIZ}/RELATORIO_entrega.html"

# Secoes que ficam no repositorio mas nao vao no PDF entregue.
# Basta o numero: o corte vai do titulo ate o proximo titulo de nivel 2 ou 3.
SECOES_FORA_DA_ENTREGA = ["6.4", "6.5"]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }

:root {
  --tinta:      #1a1a1a;
  --tinta-fraca:#555;
  --linha:      #d0d0d0;
  --fundo-th:   #f0f0f0;
  --fundo-cod:  #f6f6f6;
  --destaque:   #1f4e79;
}

* { box-sizing: border-box; }

body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: var(--tinta);
  max-width: 185mm;
  margin: 0 auto;
  padding: 12mm 8mm;
  background: #fff;
  text-align: justify;
  hyphens: auto;
}

/* ---------- titulos ---------- */

h1 {
  font-size: 19pt;
  line-height: 1.25;
  text-align: left;
  margin: 0 0 4mm 0;
  padding-bottom: 3mm;
  border-bottom: 2.5px solid var(--destaque);
  color: var(--destaque);
}

h2 {
  font-size: 14pt;
  text-align: left;
  margin: 9mm 0 3mm 0;
  padding-bottom: 1.5mm;
  border-bottom: 1px solid var(--linha);
  color: var(--destaque);
  page-break-after: avoid;
}

h3 {
  font-size: 11.5pt;
  text-align: left;
  margin: 6mm 0 2mm 0;
  color: var(--tinta);
  page-break-after: avoid;
}

/* cada secao numerada comeca em pagina nova, menos a primeira */
h2 { page-break-before: always; }
h2:first-of-type { page-break-before: avoid; }

p { margin: 0 0 2.5mm 0; orphans: 3; widows: 3; }

/* ---------- cabecalho do documento ---------- */

h1 + p {
  text-align: left;
  color: var(--tinta-fraca);
  font-size: 10pt;
  line-height: 1.7;
}

blockquote {
  margin: 4mm 0;
  padding: 2.5mm 4mm;
  border-left: 3px solid var(--destaque);
  background: #f7f9fb;
  color: var(--tinta-fraca);
  font-size: 9.5pt;
  text-align: left;
}
blockquote p { margin: 0; }

/* ---------- tabelas ---------- */

table {
  border-collapse: collapse;
  width: 100%;
  margin: 3.5mm 0 4.5mm 0;
  font-size: 9pt;
  page-break-inside: avoid;
}

thead { display: table-header-group; }

th, td {
  border: 1px solid var(--linha);
  padding: 1.6mm 2.2mm;
  text-align: left;
  vertical-align: top;
}

th {
  background: var(--fundo-th);
  font-weight: 700;
  white-space: nowrap;
}

/* colunas numericas: da segunda em diante, centralizadas */
td:not(:first-child), th:not(:first-child) { text-align: center; }

tbody tr:nth-child(even) { background: #fbfbfb; }

/* ---------- codigo ---------- */

pre {
  background: var(--fundo-cod);
  border: 1px solid var(--linha);
  border-radius: 2px;
  padding: 3mm;
  font-family: "SF Mono", "Menlo", "Consolas", monospace;
  font-size: 8.5pt;
  line-height: 1.45;
  overflow-x: auto;
  white-space: pre-wrap;
  page-break-inside: avoid;
  text-align: left;
}

code {
  font-family: "SF Mono", "Menlo", "Consolas", monospace;
  font-size: 9pt;
  background: var(--fundo-cod);
  padding: 0.3mm 1mm;
  border-radius: 2px;
}
pre code { background: none; padding: 0; font-size: inherit; }

/* ---------- listas ---------- */

ul, ol { margin: 0 0 3mm 0; padding-left: 6mm; }
li { margin-bottom: 1.2mm; }

hr { border: none; border-top: 1px solid var(--linha); margin: 6mm 0; }

strong { font-weight: 700; }

/* ---------- impressao ---------- */

@media print {
  body { padding: 0; max-width: none; }
  a { color: var(--tinta); text-decoration: none; }
}
"""


def remover_secoes(texto, numeros):
    """Tira do Markdown as secoes '### <numero> ...' indicadas.

    O corte vai do titulo ate o proximo titulo de nivel 2 ou 3 (ou o fim do
    arquivo), levando junto qualquer separador '---' que tenha ficado solto.
    """
    for numero in numeros:
        padrao = re.compile(
            r"^### " + re.escape(numero) + r" .*?(?=^#{2,3} |\Z)",
            re.MULTILINE | re.DOTALL)
        texto, n = padrao.subn("", texto)
        if n == 0:
            print(f"   AVISO: secao {numero} nao encontrada no Markdown")
    return texto


def montar_html(texto):
    """Markdown -> documento HTML completo, com o CSS de impressao embutido."""
    # As linhas "---" que separam secoes no Markdown viram <hr>; como cada <h2>
    # ja quebra pagina, esses <hr> ficariam sobrando no topo das paginas.
    texto = re.sub(r"\n---\n\n## ", "\n\n## ", texto)

    html = markdown.markdown(
        texto,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )

    titulo = "Relatório — ALPR Mercosul"
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titulo}</title>
<style>{CSS}</style>
</head>
<body>
{html}
</body>
</html>
"""


def gravar(caminho, documento, rotulo):
    open(caminho, "w", encoding="utf-8").write(documento)
    print(f"-> {os.path.basename(caminho):24s} {len(documento) // 1024:3d} KB · "
          f"{documento.count('<table>'):2d} tabelas · "
          f"{documento.count('<h2>')} seções  ({rotulo})")


def main():
    texto = open(ENTRADA, encoding="utf-8").read()

    gravar(SAIDA, montar_html(texto), "completo — repositório")
    gravar(SAIDA_ENTREGA,
           montar_html(remover_secoes(texto, SECOES_FORA_DA_ENTREGA)),
           f"sem as seções {', '.join(SECOES_FORA_DA_ENTREGA)} — entrega")

    print("\nPara gerar o PDF da entrega:")
    print("  1. abra RELATORIO_entrega.html no navegador")
    print("  2. Cmd+P -> Salvar como PDF")
    print("  3. LIGUE 'Imprimir planos de fundo' (senão as tabelas ficam sem fundo)")
    print("  4. salve na raiz do projeto e rode notebooks/11_montar_entrega.py")


if __name__ == "__main__":
    main()
