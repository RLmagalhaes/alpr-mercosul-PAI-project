# ============================================================
# Dia 5 - Pipeline fim a fim e regra do formato
# Roda no Colab via `colab exec -s dia5 -f notebooks/05_pipeline_final.py`
# IMPORTANTE: rode `colab drivemount -s dia5` (no seu terminal, interativo)
# ANTES deste script, e envie o pacote src/ pra VM uma vez por sessao:
#   colab upload src/__init__.py /content/src/__init__.py
#   colab upload src/preprocessamento.py /content/src/preprocessamento.py
#   colab upload src/validacao.py /content/src/validacao.py
#   colab upload src/metricas.py /content/src/metricas.py
#   colab upload src/pipeline.py /content/src/pipeline.py
#
# METODOLOGIA (decisao deste dia, ver DIARIO): o dataset de deteccao
# (trafficbr/vehicle-plate-color) tem fotos de veiculo inteiro, mas SEM o
# texto da placa anotado -- so a caixa da placa. Nao ha como medir
# acuracia por caractere/placa nele sem rotular a mao 60-100 placas.
# O dataset de caracteres (project-swcsj) tem cada caractere anotado
# individualmente dentro do recorte da placa -- por isso a medicao
# quantitativa (5.3/5.4) usa esse dataset: reconstroi o texto verdadeiro
# ordenando as caixas de caractere por posicao X, e roda so a parte de
# RECONHECIMENTO do pipeline (pre-processamento -> segmentacao -> CNN ->
# regra), pulando a deteccao YOLO (a imagem ja e o recorte da placa).
# A deteccao em si ja foi medida no Dia 2 (mAP50=0,992, recall=0,977).
# O pipeline COMPLETO (foto -> YOLO -> texto) e demonstrado qualitativamente
# na secao 5.5, sobre fotos do dataset de deteccao -- sem reivindicar
# acuracia quantitativa ali, porque essas fotos nao tem o texto anotado.
# ============================================================
import sys
sys.path.insert(0, "/content")
from src.preprocessamento import preparar, detectar_layout, segmentar
from src.validacao import aplicar_mascara, CLASSES
from src.metricas import (acuracia_caractere, acuracia_placa,
                          erros_por_posicao, previsao_teorica)
from src.pipeline import LeitorDePlacas

import os, glob, json
import cv2
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

RAIZ = "/content/drive/MyDrive/alpr-mercosul"
CAMINHO_DET = f"{RAIZ}/modelos/detector/weights/best.pt"
CAMINHO_CNN = f"{RAIZ}/modelos/cnn_chars.keras"

os.makedirs(f"{RAIZ}/resultados/figuras", exist_ok=True)
os.makedirs(f"{RAIZ}/resultados/tabelas", exist_ok=True)

# ---------------------------------------------------------------
# 5.1 Reconstruir o texto verdadeiro do split de teste (caracteres)
# ---------------------------------------------------------------
os.system("pip -q install roboflow")
from roboflow import Roboflow

rf = Roboflow(api_key="LgH8VW8NaRGPfLvPXv95")
DADOS_CHARS = "/content/dados/caracteres"
rf.workspace("project-swcsj").project("license-plate-character-extraction").version(2).download(
    "yolov8", location=DADOS_CHARS)

cfg = yaml.safe_load(open(f"{DADOS_CHARS}/data.yaml"))
nomes = cfg["names"] if isinstance(cfg["names"], list) else list(cfg["names"].values())


def montar_placas(split, n_esperado=7):
    """Reconstroi o texto de cada placa ordenando as caixas de caractere por X.

    Guarda tambem (xc, yc, w, h) de cada caixa -- usado na secao 5.2 pra
    recortar/desrotacionar a placa usando a geometria real das anotacoes,
    em vez de aplicar endireitar()/preparar() na foto inteira (ver DIARIO:
    essas imagens tem fundo (carroceria) e rotacao forte, muito diferentes
    do recorte limpo que sai do detector YOLO real).
    """
    registros = []
    descartadas = dict(sem_rotulo=0, tamanho_diferente_de_7=0)
    for caminho in sorted(glob.glob(f"{DADOS_CHARS}/{split}/images/*")):
        rot = caminho.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(rot):
            descartadas["sem_rotulo"] += 1
            continue
        caracteres = []
        for l in open(rot):
            partes = l.split()
            if len(partes) < 5:
                continue
            classe = str(nomes[int(partes[0])]).upper()
            if classe not in CLASSES:      # ignora EUR / "-"
                continue
            xc, yc, bw, bh = map(float, partes[1:5])
            caracteres.append((xc, yc, bw, bh, classe))
        if len(caracteres) != n_esperado:
            descartadas["tamanho_diferente_de_7"] += 1
            continue
        caracteres.sort(key=lambda t: t[0])
        texto = "".join(c[-1] for c in caracteres)
        registros.append(dict(arquivo=os.path.basename(caminho), real=texto,
                              caixas=caracteres))
    return registros, descartadas


registros_teste, descartadas_teste = montar_placas("test")
print(f"Placas de teste reconstruidas: {len(registros_teste)}")
print(f"Descartadas: {descartadas_teste}")


def recortar_via_caixas(img, caixas, margem=0.12):
    """Usa as caixas de caractere (fracoes xc,yc,w,h) pra estimar o angulo
    real da placa (reta pelos centros) e recortar so a regiao da placa,
    ja desrotacionada -- so e possivel porque este dataset anota cada
    caractere individualmente. O detector YOLO real (Dia 1/2) so devolve
    UMA caixa pra placa inteira, entao esse recurso nao existe em producao;
    la quem faz esse papel e endireitar() (heuristica sobre o contorno)."""
    h_img, w_img = img.shape[:2]
    xs = np.array([c[0] * w_img for c in caixas])
    ys = np.array([c[1] * h_img for c in caixas])
    inclinacao, _ = np.polyfit(xs, ys, 1)
    angulo = np.degrees(np.arctan(inclinacao))

    centro = (w_img / 2, h_img / 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    girada = cv2.warpAffine(img, matriz, (w_img, h_img), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)

    pts = np.stack([xs, ys, np.ones_like(xs)], axis=1)
    pts_girados = pts @ matriz.T
    larguras = np.array([c[2] * w_img for c in caixas])
    alturas = np.array([c[3] * h_img for c in caixas])

    x_min = (pts_girados[:, 0] - larguras / 2).min()
    x_max = (pts_girados[:, 0] + larguras / 2).max()
    y_min = (pts_girados[:, 1] - alturas / 2).min()
    y_max = (pts_girados[:, 1] + alturas / 2).max()

    mx, my = (x_max - x_min) * margem, (y_max - y_min) * margem
    x1, y1 = max(0, int(x_min - mx)), max(0, int(y_min - my))
    x2, y2 = min(w_img, int(x_max + mx)), min(h_img, int(y_max + my))
    return girada[y1:y2, x1:x2]


# --- Verificacao visual antes de confiar nos numeros -----------------
fig, eixos = plt.subplots(4, 3, figsize=(11, 13))
for linha, r in enumerate(registros_teste[:4]):
    caminho = f"{DADOS_CHARS}/test/images/{r['arquivo']}"
    original = cv2.imread(caminho)
    recorte = recortar_via_caixas(original, r["caixas"])
    colorida, _, realcada, binaria = preparar(recorte)
    layout, _ = detectar_layout(colorida)
    # corte_superior=0.0: recortar_via_caixas() ja entrega SO a faixa dos
    # caracteres, sem tarja. O default de 0.35 decapitava o topo de cada
    # caractere antes da segmentacao comecar (ver DIARIO, Dia 6)
    fatias = segmentar(binaria, cinza=realcada, corte_superior=0.0)
    mosaico = np.hstack(fatias)
    eixos[linha, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    eixos[linha, 0].set_title(f"original ({r['real']})", fontsize=8)
    eixos[linha, 1].imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
    eixos[linha, 1].set_title("recorte via caixas", fontsize=8)
    eixos[linha, 2].imshow(mosaico, cmap="gray")
    eixos[linha, 2].set_title("segmentado (7 fatias)", fontsize=8)
    for c in range(3):
        eixos[linha, c].axis("off")
plt.tight_layout()
plt.savefig(f"{RAIZ}/resultados/figuras/verificacao_recorte_dia5.png", dpi=140, bbox_inches="tight")
print("Figura de verificacao do recorte salva.")

# ---------------------------------------------------------------
# 5.2 Reconhecimento (sem a deteccao) sobre cada placa reconstruida
# ---------------------------------------------------------------
from tensorflow import keras
cnn = keras.models.load_model(CAMINHO_CNN)


def ler_recorte(caminho_img, caixas, usar_mascara=True):
    img = cv2.imread(caminho_img)
    if img is None:
        return None
    recorte = recortar_via_caixas(img, caixas)
    if recorte.size == 0:
        return None
    colorida, _, realcada, binaria = preparar(recorte)
    layout, _ = detectar_layout(colorida)
    # corte_superior=0.0: recortar_via_caixas() ja entrega SO a faixa dos
    # caracteres, sem tarja. O default de 0.35 decapitava o topo de cada
    # caractere antes da segmentacao comecar (ver DIARIO, Dia 6)
    fatias = segmentar(binaria, cinza=realcada, corte_superior=0.0)
    lote = np.stack(fatias).astype("float32")[..., None]
    probs = cnn.predict(lote, verbose=0)
    idx = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    bruto = "".join(CLASSES[k] for k in idx)
    final = aplicar_mascara(bruto, layout) if usar_mascara else bruto
    return bruto, final, layout, float(confs.min())


linhas = []
for r in registros_teste:
    caminho = f"{DADOS_CHARS}/test/images/{r['arquivo']}"
    resultado = ler_recorte(caminho, r["caixas"])
    if resultado is None:
        linhas.append(dict(arquivo=r["arquivo"], real=r["real"], bruto="", final="",
                           layout="", conf_minima=0.0, status="erro_leitura"))
        continue
    bruto, final, layout, conf_min = resultado
    linhas.append(dict(arquivo=r["arquivo"], real=r["real"], bruto=bruto, final=final,
                       layout=layout, conf_minima=round(conf_min, 3), status="ok"))

res = pd.DataFrame(linhas)
res.to_csv(f"{RAIZ}/resultados/tabelas/predicoes_teste.csv", index=False)

# ---------------------------------------------------------------
# 5.3 Metricas finais: CNN sozinha vs. CNN + regra do formato
# ---------------------------------------------------------------
reais = res["real"].tolist()
brutos = res["bruto"].tolist()
finais = res["final"].tolist()

acc_char_bruto = acuracia_caractere(reais, brutos)
acc_placa_bruto = acuracia_placa(reais, brutos)
acc_char_final = acuracia_caractere(reais, finais)
acc_placa_final = acuracia_placa(reais, finais)

tabela = pd.DataFrame([
    dict(versao="CNN sozinha", acc_caractere=round(acc_char_bruto, 4),
         acc_placa=round(acc_placa_bruto, 4)),
    dict(versao="CNN + regra do formato", acc_caractere=round(acc_char_final, 4),
         acc_placa=round(acc_placa_final, 4)),
])
print(tabela)
print(f"\nPrevisao teorica de acuracia por placa (erros independentes): "
     f"{previsao_teorica(acc_char_final):.4f}")
print(f"Acuracia por placa MEDIDA: {acc_placa_final:.4f} "
     f"({'acima' if acc_placa_final > previsao_teorica(acc_char_final) else 'abaixo'} "
     f"da previsao teorica)")
tabela.to_csv(f"{RAIZ}/resultados/tabelas/metricas_finais.csv", index=False)

# ---------------------------------------------------------------
# 5.4 Onde estao os erros (posicao) -- so nas placas lidas (status ok)
# ---------------------------------------------------------------
res_ok = res[res.status == "ok"]
posicoes = erros_por_posicao(res_ok["real"].tolist(), res_ok["final"].tolist())

plt.figure(figsize=(7, 3))
plt.bar(range(1, 8), posicoes)
plt.xlabel("posicao na placa"); plt.ylabel("numero de erros")
plt.title("Distribuicao dos erros por posicao (CNN + regra)")
plt.tight_layout()
plt.savefig(f"{RAIZ}/resultados/figuras/erros_por_posicao.png", dpi=140, bbox_inches="tight")
print("Figura de erros por posicao salva.")
print("Erros por posicao (1 a 7):", posicoes)

# ---------------------------------------------------------------
# 5.5 Demonstracao qualitativa do pipeline COMPLETO (foto -> texto)
# ---------------------------------------------------------------
# Sem ground truth aqui (dataset de deteccao nao anota o texto da placa) --
# serve so pra mostrar visualmente que a cadeia inteira (deteccao YOLO +
# recorte + endireitamento + segmentacao + CNN + regra) roda de ponta a
# ponta numa foto de veiculo real, nao so nos recortes ja prontos.
DADOS_DET = "/content/dados/deteccao"
rf.workspace("trafficbr").project("vehicle-plate-color").version(2).download(
    "yolov8", location=DADOS_DET)

leitor = LeitorDePlacas(CAMINHO_DET, CAMINHO_CNN)
exemplos = sorted(glob.glob(f"{DADOS_DET}/test/images/*"))[:6]

fig, eixos = plt.subplots(2, 3, figsize=(15, 8))
resultados_qualitativos = []
for ax, caminho in zip(eixos.ravel(), exemplos):
    r = leitor.ler(caminho)
    resultados_qualitativos.append(dict(arquivo=os.path.basename(caminho), **r))
    img = cv2.cvtColor(cv2.imread(caminho), cv2.COLOR_BGR2RGB)
    ax.imshow(img); ax.axis("off")
    titulo = r.get("placa_com_regra") or r["status"]
    ax.set_title(f"{titulo} ({r['status']})", fontsize=9)
plt.suptitle("Demonstracao qualitativa do pipeline completo (foto -> texto)")
plt.tight_layout()
plt.savefig(f"{RAIZ}/resultados/figuras/pipeline_exemplos.png", dpi=140, bbox_inches="tight")
print("Figura de exemplos qualitativos salva.")
print(json.dumps(resultados_qualitativos, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------
# Resumo final -> JSON no Drive
# ---------------------------------------------------------------
resumo_final = dict(
    n_placas_teste=len(res),
    n_placas_lidas_ok=int((res.status == "ok").sum()),
    descartadas_reconstrucao=descartadas_teste,
    acc_caractere_cnn_sozinha=round(acc_char_bruto, 4),
    acc_placa_cnn_sozinha=round(acc_placa_bruto, 4),
    acc_caractere_com_regra=round(acc_char_final, 4),
    acc_placa_com_regra=round(acc_placa_final, 4),
    previsao_teorica_acc_placa=round(previsao_teorica(acc_char_final), 4),
    erros_por_posicao=posicoes,
    exemplos_qualitativos=resultados_qualitativos,
)
with open(f"{RAIZ}/resultados/tabelas/resumo_dia5.json", "w") as f:
    json.dump(resumo_final, f, indent=2, ensure_ascii=False)
print("\nResumo do Dia 5 salvo em resultados/tabelas/resumo_dia5.json")
print(json.dumps({k: v for k, v in resumo_final.items()
                  if k != "exemplos_qualitativos"}, indent=2, ensure_ascii=False))
