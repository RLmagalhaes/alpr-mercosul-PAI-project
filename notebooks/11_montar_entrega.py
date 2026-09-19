"""Monta a pasta `entrega/` com tudo que vai para a professora.

Conteudo:
    entrega/
    ├── LEIA-ME.md                 instrucoes curtas (a primeira coisa a abrir)
    ├── RELATORIO.pdf              impresso a partir de RELATORIO_entrega.html
    ├── RELATORIO.html             copia de RELATORIO_entrega.html (sem 6.4/6.5)
    ├── ALPR_Mercosul.ipynb        notebook executavel, comentado
    ├── requirements.txt
    ├── src/                       modulos do sistema
    ├── modelos/                   pesos treinados (~31 MB)
    ├── imagens_exemplo/           as 30 placas do conjunto de avaliacao
    └── resultados/                tabelas (CSV) e figuras (PNG) citadas no relatorio

O notebook e montado aqui via JSON, sem depender de nbformat, e roda tanto no
Colab quanto localmente.

Uso:
    .venv/bin/python notebooks/11_montar_entrega.py
"""

import json
import os
import shutil

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTREGA = f"{RAIZ}/entrega"
BASE_IMG = f"{RAIZ}/dados/deteccao/vehicle-plates"


# ----------------------------------------------------------------------------
# 1) O notebook
# ----------------------------------------------------------------------------

def md(texto):
    return {"cell_type": "markdown", "metadata": {},
            "source": texto.strip().split("\n")}


def code(texto):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": texto.strip().split("\n")}


CELULAS = [
    md("""
# ALPR Mercosul — Reconhecimento Automático de Placas Veiculares

**Trabalho final — Processamento e Análise de Imagens (Pós-Graduação)**
Autor: Raphael Magalhães

---

## O que este sistema faz

Recebe a **foto de um veículo** e devolve o **texto da placa**:

```
foto → [YOLO detecta a placa] → [recorte + endireitamento]
     → [YOLO detecta os 7 caracteres] → [classificação]
     → [regra do formato brasileiro] → "ABC1D23"
```

## O que este notebook faz

1. Carrega os modelos treinados
2. Mostra o pipeline **passo a passo**, com figuras, numa placa real
3. Lê uma placa de ponta a ponta
4. **Reproduz a métrica principal do relatório** sobre as 30 placas de avaliação

## Como rodar

**No Google Colab:** faça o upload desta pasta inteira para o seu Drive, abra
este notebook e execute as células na ordem. A primeira célula instala o que
falta.

**Localmente:** `pip install -r requirements.txt` e execute as células.

> As seções 1 a 4 precisam apenas de `ultralytics` e rodam em CPU, em poucos
> segundos. A seção 5 (comparação com o método clássico) precisa também de
> `tensorflow` e é **opcional** — está claramente marcada.
"""),

    md("""
---
## 0. Preparação

Detecta se estamos no Colab, instala o que falta e define a pasta raiz.
Se você abriu o notebook de dentro da pasta da entrega, não precisa mudar nada.
"""),

    code("""
import os
import sys

# Estamos no Colab?
NO_COLAB = "google.colab" in sys.modules

if NO_COLAB:
    print("Ambiente: Google Colab — instalando dependências...")
    !pip -q install ultralytics
    # Se você subiu a pasta para o Drive, monte-o e ajuste RAIZ abaixo:
    # from google.colab import drive; drive.mount('/content/drive')
    # RAIZ = '/content/drive/MyDrive/entrega'
    RAIZ = os.getcwd()
else:
    print("Ambiente: local")
    RAIZ = os.getcwd()

# Se o notebook estiver numa subpasta, sobe até achar a pasta 'modelos'
while not os.path.isdir(f"{RAIZ}/modelos") and RAIZ != "/":
    RAIZ = os.path.dirname(RAIZ)

sys.path.insert(0, RAIZ)
print("Pasta raiz :", RAIZ)
print("Conteúdo   :", sorted(os.listdir(RAIZ)))
"""),

    md("""
---
## 1. Carregar os modelos

Três modelos foram treinados neste trabalho:

| Modelo | Arquivo | Papel |
| --- | --- | --- |
| YOLO11n — detector de placas | `detector_best.pt` | acha a placa na foto do veículo |
| YOLO11n — detector de caracteres | `chars_best.pt` | acha e classifica os 7 caracteres |
| CNN de 36 classes | `cnn_chars_compat.keras` | classificador alternativo (seção 5) |
"""),

    code("""
from ultralytics import YOLO

DETECTOR = f"{RAIZ}/modelos/detector_best.pt"
CHARS    = f"{RAIZ}/modelos/chars_best.pt"
CNN      = f"{RAIZ}/modelos/cnn_chars_compat.keras"

detector = YOLO(DETECTOR)
chars    = YOLO(CHARS)

print("Detector de placas    :", len(detector.names), "classe(s) ->", detector.names)
print("Detector de caracteres:", len(chars.names), "classes")
print("\\nModelos carregados com sucesso.")
"""),

    md("""
---
## 2. O pipeline, passo a passo

Aqui cada etapa aparece separadamente, para ficar visível o que o sistema faz
com a imagem antes de produzir o texto.
"""),

    code("""
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Escolha aqui qual das 30 imagens de exemplo usar (0 a 29)
INDICE = 0

imagens = sorted(glob.glob(f"{RAIZ}/imagens_exemplo/*.jpg"))
caminho = imagens[INDICE]
img = cv2.imread(caminho)
print(f"Imagem {INDICE+1} de {len(imagens)}: {os.path.basename(caminho)}")
print(f"Dimensões: {img.shape[1]}x{img.shape[0]} px")
"""),

    code("""
# ---- ETAPA 1: detectar a placa ----------------------------------------------
# O detector devolve as caixas candidatas; ficamos com a de maior confiança.

resultado = detector.predict(img, conf=0.25, verbose=False)[0]
caixas = resultado.boxes

i = int(np.argmax(caixas.conf.cpu().numpy()))
x1, y1, x2, y2 = caixas.xyxy.cpu().numpy()[i].astype(int)
conf_placa = float(caixas.conf.cpu().numpy()[i])

area_pct = ((x2 - x1) * (y2 - y1)) / (img.shape[0] * img.shape[1]) * 100
print(f"Placa encontrada com confiança {conf_placa:.3f}")
print(f"Caixa: ({x1}, {y1}) -> ({x2}, {y2})")
print(f"A placa ocupa {area_pct:.2f}% da imagem  <- a dificuldade central do problema")
"""),

    code("""
# ---- ETAPA 2: recortar com margem de 8% -------------------------------------
# A margem evita cortar caracteres colados na borda da caixa detectada.

from src.preprocessamento import recortar

placa = recortar(img, (x1, y1, x2, y2), margem=0.08)
print("Recorte da placa:", placa.shape[1], "x", placa.shape[0], "px")
"""),

    code("""
# ---- ETAPA 3: detectar os 7 caracteres --------------------------------------
# conf=0.05 (e não o padrão 0.25) porque a placa tem exatamente 7 caracteres:
# ficamos com os 7 mais confiantes, então falso positivo é barato e faltar
# caractere é caro. A escolha está medida na seção 5.5 do relatório.

deteccoes = chars.predict(placa, conf=0.05, verbose=False)[0]
print(f"Caracteres detectados (antes de filtrar): {len(deteccoes.boxes)}")

achados = []
for caixa, cls, cf in zip(deteccoes.boxes.xyxy.cpu().numpy(),
                          deteccoes.boxes.cls.cpu().numpy().astype(int),
                          deteccoes.boxes.conf.cpu().numpy()):
    classe = str(chars.names[int(cls)]).upper()
    if classe in {"EUR", "-"}:        # não são caracteres de placa
        continue
    achados.append((caixa, classe, float(cf)))

achados.sort(key=lambda t: -t[2])      # os 7 mais confiantes
achados = achados[:7]
achados.sort(key=lambda t: t[0][0])    # reordenados da esquerda para a direita

print("Leitura bruta:", "".join(c[1] for c in achados))
"""),

    code("""
# ---- Visualização das três etapas -------------------------------------------

fig, eixos = plt.subplots(1, 3, figsize=(16, 4.5))

# (a) a foto com a placa marcada
vis = img.copy()
cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
eixos[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
eixos[0].set_title(f"1. Placa detectada (conf {conf_placa:.2f})\\n"
                   f"ocupa {area_pct:.2f}% da imagem")

# (b) o recorte
eixos[1].imshow(cv2.cvtColor(placa, cv2.COLOR_BGR2RGB))
eixos[1].set_title("2. Recorte com margem de 8%")

# (c) os caracteres localizados
vis_chars = placa.copy()
for (cx1, cy1, cx2, cy2), classe, cf in achados:
    cv2.rectangle(vis_chars, (int(cx1), int(cy1)), (int(cx2), int(cy2)),
                  (0, 0, 255), 1)
    cv2.putText(vis_chars, classe, (int(cx1), max(12, int(cy1) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
eixos[2].imshow(cv2.cvtColor(vis_chars, cv2.COLOR_BGR2RGB))
eixos[2].set_title(f"3. Caracteres localizados\\n{''.join(c[1] for c in achados)}")

for e in eixos:
    e.axis("off")
plt.tight_layout()
plt.show()
"""),

    md("""
---
## 3. A regra do formato

O Brasil tem dois padrões de placa, ambos com 7 caracteres, mas com **tipagem
diferente por posição**:

| Padrão | Máscara | Exemplo |
| --- | --- | --- |
| Mercosul | 3 letras · 1 dígito · 1 letra · 2 dígitos (`LLLDLDD`) | `ABC1D23` |
| Antigo | 3 letras · 4 dígitos (`LLLDDDD`) | `ABC1234` |

Sabendo o layout (detectado pela **tarja azul** da placa), um caractere
impossível para a posição pode ser trocado pelo equivalente visual — sem
treinar nada a mais. São 15 linhas derivadas da legislação.

No relatório isso rende **+10 pontos** de acurácia por caractere.
"""),

    code("""
from src.preprocessamento import detectar_layout
from src.validacao import aplicar_mascara

bruto = "".join(c[1] for c in achados)
layout, score_azul = detectar_layout(placa)
final = aplicar_mascara(bruto, layout)

print(f"Layout detectado : {layout}  (score de azul: {score_azul:.3f})")
print(f"Leitura bruta    : {bruto}")
print(f"Após a regra     : {final}")
if bruto != final:
    trocas = [f"pos.{i+1}: {a} -> {b}"
              for i, (a, b) in enumerate(zip(bruto, final)) if a != b]
    print(f"Correções        : {', '.join(trocas)}")
else:
    print("A regra não precisou corrigir nada nesta placa.")
"""),

    md("""
---
## 4. Leitura completa, de ponta a ponta

Tudo o que foi feito acima está encapsulado na classe `LeitorDePlacasYOLO`.
Uma chamada, um dicionário de resposta.
"""),

    code("""
from src.leitor_yolo import LeitorDePlacasYOLO

leitor = LeitorDePlacasYOLO(DETECTOR, CHARS, CNN)

resposta = leitor.ler(caminho, modo="yolo")
for chave, valor in resposta.items():
    print(f"{chave:20s}: {valor}")
"""),

    md("""
### Leitura das primeiras 10 placas do conjunto de avaliação

A coluna `real` é o gabarito transcrito e conferido por humano.
"""),

    code("""
import pandas as pd

gabarito = pd.read_csv(f"{RAIZ}/resultados/tabelas/gabarito_mercosul.csv", dtype=str)
gabarito["texto"] = gabarito["texto"].str.strip().str.upper()

linhas = []
for _, r in gabarito.head(10).iterrows():
    res = leitor.ler(f"{RAIZ}/imagens_exemplo/{r['arquivo']}", modo="yolo")
    previsto = res.get("placa_com_regra", "") or ""
    linhas.append(dict(
        real=r["texto"],
        previsto=previsto,
        layout=res.get("layout", ""),
        acertou="SIM" if previsto == r["texto"] else "não",
        chars_certos=sum(1 for a, b in zip(r["texto"], previsto) if a == b)))

print(pd.DataFrame(linhas).to_string(index=False))
"""),

    md("""
---
## 5. Reprodução do resultado principal *(opcional — requer TensorFlow)*

Esta seção reproduz a **Tabela 5.4 do relatório**: a comparação dos três
pipelines sobre as mesmas 30 placas.

O experimento isola a variável: o pipeline **B** usa **a mesma CNN** do
pipeline A e troca **apenas** a forma de localizar os caracteres. É isso que
demonstra que o gargalo do sistema estava na segmentação.

> ⚠️ Leva alguns minutos em CPU e precisa de `tensorflow`.
> Pode ser pulada sem prejuízo — os resultados já estão em
> `resultados/tabelas/comparacao_pipelines.csv`.
"""),

    code("""
# Descomente a linha abaixo para instalar o TensorFlow no Colab, se necessário
# !pip -q install tensorflow

RODAR_COMPARACAO = False    # <- mude para True para executar de verdade

if RODAR_COMPARACAO:
    from src.metricas import acuracia_caractere, acuracia_placa
    from src.pipeline import LeitorDePlacas

    classico = LeitorDePlacas(DETECTOR, CNN)
    reais, a_cols, b_cols, c_cols = [], [], [], []

    for _, r in gabarito.iterrows():
        p = f"{RAIZ}/imagens_exemplo/{r['arquivo']}"
        reais.append(r["texto"])
        a_cols.append(classico.ler(p).get("placa_com_regra", "") or "")
        b_cols.append(leitor.ler(p, modo="cnn").get("placa_com_regra", "") or "")
        c_cols.append(leitor.ler(p, modo="yolo").get("placa_com_regra", "") or "")

    tabela = pd.DataFrame([
        dict(pipeline="A) clássico + CNN",
             acc_caractere=round(acuracia_caractere(reais, a_cols), 4),
             acc_placa=round(acuracia_placa(reais, a_cols), 4)),
        dict(pipeline="B) YOLO caixas + mesma CNN",
             acc_caractere=round(acuracia_caractere(reais, b_cols), 4),
             acc_placa=round(acuracia_placa(reais, b_cols), 4)),
        dict(pipeline="C) YOLO caixa + classe",
             acc_caractere=round(acuracia_caractere(reais, c_cols), 4),
             acc_placa=round(acuracia_placa(reais, c_cols), 4)),
    ])
    print(tabela.to_string(index=False))
else:
    print("Resultado já medido (resultados/tabelas/comparacao_pipelines.csv):\\n")
    print(pd.read_csv(f"{RAIZ}/resultados/tabelas/comparacao_pipelines.csv")
          .to_string(index=False))
"""),

    md("""
---
## 6. Resumo dos resultados

| Métrica | Meta | Obtido |
| --- | --- | --- |
| mAP@0.5 — detecção da placa | > 0,90 | **0,992** ✅ |
| mAP@0.5 — detecção de caracteres (teste) | — | **0,9305** |
| Acurácia por caractere — CNN isolada | > 0,95 | 0,9435 ⚠️ |
| Acurácia por caractere — fim a fim | > 0,95 | **0,7190** ❌ |
| Acurácia por placa — fim a fim | > 0,80 | **0,3333** ❌ |

**Duas metas não foram atingidas.** O sistema lê corretamente 10 das 30 placas.

O que o trabalho oferece no lugar é um **diagnóstico medido**: o experimento
controlado da seção 5 mostra que, mantendo a CNN idêntica e trocando apenas a
localização dos caracteres, a acurácia por caractere **dobra** (0,300 → 0,619) e
a acurácia por placa sai de zero. O gargalo era a segmentação — não a CNN, não a
base de dados, não o detector de placas.

A discussão completa, com as limitações e a análise de erros, está no
**`RELATORIO.pdf`**.
"""),
]

NOTEBOOK = {
    "cells": CELULAS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}


# ----------------------------------------------------------------------------
# 2) O LEIA-ME
# ----------------------------------------------------------------------------

LEIAME = """# ALPR Mercosul — Entrega

**Trabalho final — Processamento e Análise de Imagens (Pós-Graduação)**
Autor: Raphael Magalhães

Sistema que recebe a foto de um veículo e devolve o texto da placa.

---

## Por onde começar

| Arquivo | O que é |
| --- | --- |
| **`RELATORIO.pdf`** | O relatório completo. **Comece por aqui.** |
| **`ALPR_Mercosul.ipynb`** | Notebook executável: o sistema funcionando, passo a passo |

## Rodar o notebook

**Opção 1 — Google Colab (mais simples, não exige instalar nada):**
faça o upload desta pasta para o Drive, abra `ALPR_Mercosul.ipynb` no Colab e
execute as células na ordem. A primeira célula instala o que falta.

**Opção 2 — localmente:**

```bash
pip install -r requirements.txt
jupyter notebook ALPR_Mercosul.ipynb
```

As seções 1 a 4 rodam em CPU, em segundos, e precisam apenas de `ultralytics`.
A seção 5 é opcional e requer `tensorflow`.

## O que tem nesta pasta

```
RELATORIO.pdf          relatório completo
RELATORIO.html         fonte do PDF
ALPR_Mercosul.ipynb    notebook executável
src/                   módulos do sistema (o código de verdade)
modelos/               os 3 modelos treinados
imagens_exemplo/       as 30 placas do conjunto de avaliação
resultados/tabelas/    todos os números do relatório, em CSV
resultados/figuras/    todas as figuras do relatório, em PNG
requirements.txt       dependências com versões fixas
```

## Resultados em uma tabela

| Métrica | Meta | Obtido |
| --- | --- | --- |
| mAP@0.5 — detecção da placa | > 0,90 | **0,992** |
| mAP@0.5 — detecção de caracteres | — | **0,9305** |
| Acurácia por caractere — CNN isolada | > 0,95 | 0,9435 |
| Acurácia por caractere — fim a fim | > 0,95 | **0,7190** |
| Acurácia por placa — fim a fim | > 0,80 | **0,3333** |

Duas metas não foram atingidas. O relatório discute por quê, e traz o
experimento controlado que identifica onde o desempenho se perde.

## Privacidade

Placa veicular identifica indiretamente uma pessoa. Este é um exercício
acadêmico, sem finalidade de vigilância. Foram usadas apenas bases públicas
licenciadas para pesquisa; o sistema produz apenas a string da placa e não
consulta, armazena nem cruza qualquer dado sobre proprietário ou veículo.
Um sistema com esta acurácia não deve ser usado para decisões automáticas
sobre pessoas — por isso ele recusa a leitura (`revisao_manual`) em vez de
arriscar um palpite quando a confiança fica baixa.
"""

REQUISITOS = """# Dependências da entrega — versões fixas, as mesmas usadas na avaliação.
# Seções 1 a 4 do notebook precisam apenas das quatro primeiras linhas.
ultralytics==8.3.0
opencv-python==4.10.0.84
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.2
# Apenas para a seção 5 (comparação com o método clássico):
tensorflow==2.17.0
"""


# ----------------------------------------------------------------------------
# 3) Montagem
# ----------------------------------------------------------------------------

def achar_pdf():
    """Devolve o PDF do relatorio na raiz, seja qual for o nome do arquivo.

    O PDF e impresso a mao pelo navegador, que costuma salvar com o titulo da
    pagina ("Relatorio - ALPR Mercosul.pdf") em vez de "RELATORIO.pdf". Em vez
    de exigir um nome exato, varremos a raiz e preferimos o que parecer ser o
    relatorio; havendo duvida, fica o mais recente.
    """
    pdfs = [f"{RAIZ}/{a}" for a in os.listdir(RAIZ) if a.lower().endswith(".pdf")]
    if not pdfs:
        return None
    provaveis = [p for p in pdfs
                 if "relat" in os.path.basename(p).lower()[:6]]
    return max(provaveis or pdfs, key=os.path.getmtime)


def copiar_imagens_do_gabarito():
    """Copia as 30 imagens do conjunto de avaliacao para a pasta da entrega."""
    gab = pd.read_csv(f"{RAIZ}/resultados/tabelas/gabarito_mercosul.csv", dtype=str)
    destino = f"{ENTREGA}/imagens_exemplo"
    os.makedirs(destino, exist_ok=True)
    copiadas, faltando = 0, []
    for _, r in gab.iterrows():
        origem = f"{BASE_IMG}/{r['split']}/images/{r['arquivo']}"
        if os.path.exists(origem):
            shutil.copy2(origem, f"{destino}/{r['arquivo']}")
            copiadas += 1
        else:
            faltando.append(r["arquivo"])
    return copiadas, faltando


def tamanho(caminho):
    total = sum(os.path.getsize(os.path.join(p, f))
                for p, _, arqs in os.walk(caminho) for f in arqs)
    return total / 1024 / 1024


def main():
    if os.path.exists(ENTREGA):
        shutil.rmtree(ENTREGA)
    os.makedirs(ENTREGA)

    # --- notebook, leia-me, requirements ---
    with open(f"{ENTREGA}/ALPR_Mercosul.ipynb", "w", encoding="utf-8") as f:
        json.dump(NOTEBOOK, f, ensure_ascii=False, indent=1)
    open(f"{ENTREGA}/LEIA-ME.md", "w", encoding="utf-8").write(LEIAME)
    open(f"{ENTREGA}/requirements.txt", "w", encoding="utf-8").write(REQUISITOS)

    # --- relatorio ---
    # A entrega leva a versao SEM as secoes 6.4 e 6.5 (ver notebooks/10_*.py).
    html_entrega = f"{RAIZ}/RELATORIO_entrega.html"
    if not os.path.exists(html_entrega):
        html_entrega = f"{RAIZ}/RELATORIO.html"
        print("  AVISO: RELATORIO_entrega.html nao existe — copiando a versao "
              "completa. Rode notebooks/10_gerar_html_relatorio.py antes.")
    shutil.copy2(html_entrega, f"{ENTREGA}/RELATORIO.html")

    pdf = achar_pdf()
    if pdf:
        shutil.copy2(pdf, f"{ENTREGA}/RELATORIO.pdf")
        if os.path.getmtime(pdf) < os.path.getmtime(html_entrega):
            print(f"  NOTA: '{os.path.basename(pdf)}' e mais antigo que o HTML da"
                  " entrega. Pode ser so o HTML ter sido regerado sem mudanca de"
                  " conteudo; confira as secoes do PDF antes de entregar.")

    # --- codigo ---
    shutil.copytree(f"{RAIZ}/src", f"{ENTREGA}/src",
                    ignore=shutil.ignore_patterns("__pycache__"))

    # --- modelos (so os 3 que o notebook usa) ---
    os.makedirs(f"{ENTREGA}/modelos")
    for m in ["detector_best.pt", "chars_best.pt", "cnn_chars_compat.keras"]:
        shutil.copy2(f"{RAIZ}/modelos/{m}", f"{ENTREGA}/modelos/{m}")

    # --- resultados ---
    shutil.copytree(f"{RAIZ}/resultados", f"{ENTREGA}/resultados")

    # --- imagens ---
    copiadas, faltando = copiar_imagens_do_gabarito()

    # --- relatorio da montagem ---
    print(f"Pasta de entrega montada em: {ENTREGA}\n")
    for pasta in ["", "src", "modelos", "imagens_exemplo",
                  "resultados/tabelas", "resultados/figuras"]:
        caminho = f"{ENTREGA}/{pasta}".rstrip("/")
        arquivos = [a for a in os.listdir(caminho)
                    if os.path.isfile(os.path.join(caminho, a))]
        nome = pasta or "(raiz)"
        mb = f"{tamanho(caminho):6.1f} MB" if pasta else ""
        print(f"  {nome:22s} {len(arquivos):3d} arquivo(s) {mb}")

    print(f"\n  imagens do gabarito copiadas: {copiadas}/30")
    if faltando:
        print(f"  ATENCAO — nao encontradas: {faltando}")
    print(f"\n  TAMANHO TOTAL: {tamanho(ENTREGA):.1f} MB")

    if not os.path.exists(f"{ENTREGA}/RELATORIO.pdf"):
        print("\n  PENDENTE: gerar RELATORIO.pdf a partir do RELATORIO.html")
        print("            (abrir no navegador -> Cmd+P -> Salvar como PDF)")
        print("            e rodar este script de novo para incluí-lo.")


if __name__ == "__main__":
    main()
