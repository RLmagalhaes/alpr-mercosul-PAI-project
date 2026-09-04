# ============================================================
# Dia 3 - Pre-processamento e dataset de caracteres
# Roda no Colab via `colab exec -s dia3 -f notebooks/03_preprocessamento_dataset.py`
# IMPORTANTE: rode `colab drivemount -s dia3` (no seu terminal, interativo)
# ANTES deste script.
#
# As funcoes de tratamento (endireitar, preparar, detectar_layout,
# projecao_vertical, segmentar) sao as REAIS de src/preprocessamento.py --
# nao redefinidas aqui. Pra isso, envie o pacote src/ pra VM uma vez por
# sessao:
#   colab upload src/__init__.py /content/src/__init__.py
#   colab upload src/preprocessamento.py /content/src/preprocessamento.py
# ============================================================
import sys
sys.path.insert(0, "/content")
from src.preprocessamento import (endireitar, preparar, detectar_layout,
                                   projecao_vertical, segmentar)

import os, glob, cv2, json
import matplotlib.pyplot as plt
import pandas as pd
import yaml

RAIZ = "/content/drive/MyDrive/alpr-mercosul"
DADOS = "/content/dados/deteccao"
DESTINO_PLACAS = "/content/dados/placas_recortadas"

# --- 3.0 Recria as placas recortadas do Dia 2 --------------------
# Nao persistem no Drive de proposito (ver DIARIO, Dia 2) -- e rapido de
# regerar (so I/O + cv2, sem GPU), entao roda de novo no inicio desta sessao.
os.system("pip -q install roboflow")
from roboflow import Roboflow

rf = Roboflow(api_key="LgH8VW8NaRGPfLvPXv95")
rf.workspace("trafficbr").project("vehicle-plate-color").version(2).download(
    "yolov8", location=DADOS)


def parse_anotacao(partes):
    valores = list(map(float, partes[1:]))
    if len(valores) == 4:
        return tuple(valores)
    if len(valores) >= 6 and len(valores) % 2 == 0:
        xs, ys = valores[0::2], valores[1::2]
        return ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2,
                 max(xs)-min(xs), max(ys)-min(ys))
    return None


def recortar_todas(split, margem=0.08, largura_min=30, altura_min=10):
    destino = f"{DESTINO_PLACAS}/{split}"
    os.makedirs(destino, exist_ok=True)
    n = 0
    for caminho in sorted(glob.glob(f"{DADOS}/{split}/images/*")):
        img = cv2.imread(caminho)
        if img is None:
            continue
        h, w = img.shape[:2]
        rot = caminho.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(rot):
            continue
        base = os.path.splitext(os.path.basename(caminho))[0]
        for i, l in enumerate(open(rot)):
            partes = l.split()
            if len(partes) < 5:
                continue
            anotacao = parse_anotacao(partes)
            if anotacao is None:
                continue
            xc, yc, bw, bh = anotacao
            x1, y1 = (xc - bw/2) * w, (yc - bh/2) * h
            x2, y2 = (xc + bw/2) * w, (yc + bh/2) * h
            mx, my = (x2 - x1) * margem, (y2 - y1) * margem
            x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
            recorte = img[y1:y2, x1:x2]
            if recorte.size == 0 or recorte.shape[0] < altura_min or recorte.shape[1] < largura_min:
                continue
            cv2.imwrite(f"{destino}/{base}_{i}.jpg", recorte)
            n += 1
    return n


for s in ["train", "valid", "test"]:
    print(s, "->", recortar_todas(s), "placas recortadas")

# ---------------------------------------------------------------
# 3.1 / 3.2 Efeito do pre-processamento numa placa bem alinhada
# ---------------------------------------------------------------
# Escolhido a dedo entre os poucos recortes de tamanho razoavel do teste
# (ver Dia 2: 76% dos recortes sao muito pequenos). Este exemplo especifico
# (images160_jpg) fica reto o suficiente pra mostrar o pipeline funcionando
# bem -- o caso torto (limitacao) fica documentado em 3.3.
caso_bom = sorted(glob.glob(f"{DESTINO_PLACAS}/test/images160_jpg*"))[0]
exemplo = endireitar(cv2.imread(caso_bom))
p, cinza, realcada, binaria = preparar(exemplo)
layout, score_azul = detectar_layout(p)
print("Layout detectado:", layout, "| score azul:", score_azul)

fig, eixos = plt.subplots(1, 4, figsize=(16, 3))
for ax, img, titulo in zip(eixos,
        [cv2.cvtColor(p, cv2.COLOR_BGR2RGB), cinza, realcada, binaria],
        ["recortada", "escala de cinza", "CLAHE", "binarizada (Otsu)"]):
    ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
    ax.set_title(titulo, fontsize=10); ax.axis("off")
plt.suptitle(f"Layout detectado: {layout} (score azul = {score_azul})")
plt.tight_layout()
plt.savefig(f"{RAIZ}/resultados/figuras/preprocessamento.png", dpi=140, bbox_inches="tight")
print("Figura de pre-processamento salva.")

# Resultado obtido: layout = mercosul (score azul = 0.252), placa legivel em
# todas as etapas.

# ---------------------------------------------------------------
# 3.3 Segmentar os 7 caracteres -- caso bom vs. caso com perspectiva forte
# ---------------------------------------------------------------
# PROBLEMA DE DADOS REGISTRADO (nao contornado): endireitar() so corrige
# ROTACAO no plano (via minAreaRect), nao PERSPECTIVA. Fotos tiradas de
# angulo (camera baixa/lateral olhando pra cima) tem distorcao de
# perspectiva, nao so rotacao -- a segmentacao por projecao (que assume
# fatias de largura igual) falha nesses casos, como o exemplo abaixo mostra.
# Corrigir perspectiva de verdade exigiria os 4 cantos da placa anotados,
# que este dataset nao tem -- por isso a limitacao fica documentada, nao
# mascarada com uma heuristica melhor.
caso_torto = [c for c in sorted(glob.glob(f"{DESTINO_PLACAS}/test/*.jpg"))
              if "306db6d3" in c][0]

fig, eixos = plt.subplots(2, 8, figsize=(16, 5))
for linha, (caminho, rotulo) in enumerate([(caso_bom, "caso bem alinhado"),
                                            (caso_torto, "caso com perspectiva forte")]):
    img = endireitar(cv2.imread(caminho))
    p2, _, _, binaria2 = preparar(img)
    recortes = segmentar(binaria2)
    eixos[linha, 0].imshow(cv2.cvtColor(p2, cv2.COLOR_BGR2RGB))
    eixos[linha, 0].set_title(rotulo, fontsize=8); eixos[linha, 0].axis("off")
    for i, r in enumerate(recortes):
        eixos[linha, i + 1].imshow(r, cmap="gray"); eixos[linha, i + 1].axis("off")
plt.suptitle("Segmentacao em 7 fatias: placa bem alinhada vs. com perspectiva forte")
plt.tight_layout()
plt.savefig(f"{RAIZ}/resultados/figuras/segmentacao.png", dpi=140, bbox_inches="tight")
print("Figura de segmentacao (comparativa) salva.")

# Resultado obtido: caso bem alinhado -> 7 fatias limpas e legiveis
# (LQQ4J66, com leve sangramento do caractere vizinho, esperado por design).
# Caso com perspectiva forte -> fatias completamente desalinhadas com os
# caracteres reais (NYSGA35 fica ilegivel na segmentacao). Efeito esperado:
# a taxa de erro do pipeline completo (Dia 5) deve ser bem maior nesse tipo
# de foto.

# ---------------------------------------------------------------
# 3.4 Gerar o dataset de caracteres
# ---------------------------------------------------------------
# Dataset SEPARADO do de deteccao: project-swcsj/license-plate-character-
# extraction v2 (36 classes 0-9/A-Z + 2 extras que sao descartadas: "-" e
# "EUR"). Cada imagem ja e um recorte de placa com os caracteres anotados
# individualmente -- por isso nao depende do segmentar() nem dos recortes
# do Dia 2 pra treino, so pra inferencia (Dia 5).
DADOS_CHARS = "/content/dados/caracteres"
rf.workspace("project-swcsj").project("license-plate-character-extraction").version(2).download(
    "yolov8", location=DADOS_CHARS)

# PROBLEMA DE DADOS REGISTRADO (pendencia do Dia 0, resolvida aqui): o
# README do Roboflow confirma "Resize to 640x640 (Stretch)" -- as imagens
# foram esticadas sem preservar a proporcao original, o que deforma os
# caracteres de forma nao uniforme (a proporcao original nao fica
# registrada em lugar nenhum pra desfazer). Aceito como limitacao do
# dataset escolhido -- nao da pra reconstruir a proporcao original sem
# inventar informacao.
print(open(f"{DADOS_CHARS}/README.roboflow.txt").read()[:600])

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
cfg = yaml.safe_load(open(f"{DADOS_CHARS}/data.yaml"))
nomes = cfg["names"] if isinstance(cfg["names"], list) else list(cfg["names"].values())
DESTINO_CHARS = "/content/dados/chars"


def gerar_chars(split, saida=(32, 32)):
    base_destino = f"{DESTINO_CHARS}/{split}"
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    c = dict(ok=0, classe_ignorada=0, imagem_ilegivel=0, sem_rotulo=0,
             anotacao_invalida=0, recorte_pequeno_demais=0)
    ignoradas = {}
    for caminho in sorted(glob.glob(f"{DADOS_CHARS}/{split}/images/*")):
        img = cv2.imread(caminho)
        if img is None:
            c["imagem_ilegivel"] += 1
            continue
        h, w = img.shape[:2]
        rot = caminho.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(rot):
            c["sem_rotulo"] += 1
            continue
        nome_base = os.path.splitext(os.path.basename(caminho))[0]
        for i, l in enumerate(open(rot)):
            partes = l.split()
            if len(partes) < 5:
                c["anotacao_invalida"] += 1
                continue
            classe = str(nomes[int(partes[0])]).upper()
            if classe not in CLASSES:
                c["classe_ignorada"] += 1
                ignoradas[classe] = ignoradas.get(classe, 0) + 1
                continue
            xc, yc, bw, bh = map(float, partes[1:5])
            x1, y1 = max(0, int((xc-bw/2)*w)), max(0, int((yc-bh/2)*h))
            x2, y2 = min(w, int((xc+bw/2)*w)), min(h, int((yc+bh/2)*h))
            rec = img[y1:y2, x1:x2]
            if rec.size == 0 or rec.shape[0] < 8 or rec.shape[1] < 5:
                c["recorte_pequeno_demais"] += 1
                continue
            cinza = clahe.apply(cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY))
            binaria = cv2.threshold(cinza, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            destino = f"{base_destino}/{classe}"
            os.makedirs(destino, exist_ok=True)
            cv2.imwrite(f"{destino}/{nome_base}_{i}.png",
                        cv2.resize(binaria, saida, interpolation=cv2.INTER_AREA))
            c["ok"] += 1
    c["classes_ignoradas_detalhe"] = ignoradas
    return c


resumo_chars = {s: gerar_chars(s) for s in ["train", "valid", "test"]}
for s, c in resumo_chars.items():
    print(s, c)

os.makedirs(f"{RAIZ}/resultados/tabelas", exist_ok=True)
with open(f"{RAIZ}/resultados/tabelas/chars_geracao_resumo.json", "w") as f:
    json.dump(resumo_chars, f, indent=2, ensure_ascii=False)

# Resultado obtido (rodado em 03/09):
#   train: ok=30530, classe_ignorada=1695 (EUR=1652, "-"=43)
#   valid: ok=958,  classe_ignorada=44   (EUR=43, "-"=1)
#   test:  ok=230,  classe_ignorada=13   (EUR=13)
#   Total: 31718 caracteres rotulados.

# --- Balanceamento das classes -----------------------------------
contagem = {c: len(glob.glob(f"{DESTINO_CHARS}/train/{c}/*"))
            for c in CLASSES if os.path.isdir(f"{DESTINO_CHARS}/train/{c}")}
serie = pd.Series(contagem).sort_values()
print("Classes com menos exemplos (treino):\n", serie.head(10))
serie.to_csv(f"{RAIZ}/resultados/tabelas/chars_balanceamento_treino.csv")

plt.figure(figsize=(14, 3))
serie.sort_index().plot.bar()
plt.title("Exemplos por classe (treino)")
plt.tight_layout()
plt.savefig(f"{RAIZ}/resultados/figuras/chars_balanceamento.png", dpi=140, bbox_inches="tight")
print("Figura de balanceamento salva.")

# Resultado obtido: forte desbalanceamento -- Q (120) e O (228) muito
# abaixo do resto (a maioria das letras fica entre 400-1500, digitos entre
# 1000-2100). Q e O tambem sao visualmente parecidos com 0, o que soma
# risco de confusao no Dia 4 -- acompanhar de perto na matriz de confusao.

# PROBLEMA DE DADOS REGISTRADO (nao contornado): checagem visual de uma
# amostra aleatoria de cada classe encontrou pelo menos um exemplo rotulado
# como "O" que na verdade mostra um "H" -- ruido de anotacao do dataset
# original (comum em datasets anotados em massa). Aceito como parte do
# ruido de rotulo esperado, sem tentar filtrar manualmente (nao ha tempo
# nem um metodo confiavel pra achar todos os casos parecidos sem revisar
# as ~31 mil imagens uma a uma).
