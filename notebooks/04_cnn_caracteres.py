# ============================================================
# Dia 4 - CNN de caracteres (36 classes)
# Roda no Colab via `colab exec -s dia4 -f notebooks/04_cnn_caracteres.py`
# IMPORTANTE: rode `colab drivemount -s dia4` (no seu terminal, interativo)
# ANTES deste script.
#
# O disco local do Colab (/content) e apagado a cada sessao nova (ver
# DIARIO, Dia 1/2/3) -- por isso a secao 4.0 regenera `chars/` do zero
# a partir do dataset de caracteres anotados (project-swcsj), com o MESMO
# codigo usado no Dia 3 (so ida rapida de I/O + cv2, sem GPU). Isso garante
# que os numeros batem com os do Dia 3 (31.718 caracteres, 36 classes).
# ============================================================
import os, glob, json
import cv2
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

RAIZ = "/content/drive/MyDrive/alpr-mercosul"
DADOS_CHARS = "/content/dados/caracteres"
DESTINO_CHARS = "/content/dados/chars"

# --- 4.0 Regenera chars/ (identico ao Dia 3) ---------------------
os.system("pip -q install roboflow")
from roboflow import Roboflow

rf = Roboflow(api_key="LgH8VW8NaRGPfLvPXv95")
rf.workspace("project-swcsj").project("license-plate-character-extraction").version(2).download(
    "yolov8", location=DADOS_CHARS)

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
cfg = yaml.safe_load(open(f"{DADOS_CHARS}/data.yaml"))
nomes = cfg["names"] if isinstance(cfg["names"], list) else list(cfg["names"].values())


def gerar_chars(split, saida=(32, 32)):
    base_destino = f"{DESTINO_CHARS}/{split}"
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    c = dict(ok=0, classe_ignorada=0, imagem_ilegivel=0, sem_rotulo=0,
             anotacao_invalida=0, recorte_pequeno_demais=0)
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
    return c


resumo_chars = {s: gerar_chars(s) for s in ["train", "valid", "test"]}
for s, c in resumo_chars.items():
    print(s, c)

# ---------------------------------------------------------------
# 4.1 Carregar os dados
# ---------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

TAM, LOTE = (32, 32), 128


def carregar(split, embaralhar):
    return keras.utils.image_dataset_from_directory(
        f"{DESTINO_CHARS}/{split}",
        labels="inferred", label_mode="int", color_mode="grayscale",
        image_size=TAM, batch_size=LOTE, shuffle=embaralhar, seed=42)


treino = carregar("train", True)
val    = carregar("valid", False)
teste  = carregar("test",  False)

CLASSES_MODELO = treino.class_names
print(f"{len(CLASSES_MODELO)} classes:", CLASSES_MODELO)

AUTO = tf.data.AUTOTUNE
treino = treino.cache().prefetch(AUTO)
val    = val.cache().prefetch(AUTO)
teste  = teste.cache().prefetch(AUTO)

# ---------------------------------------------------------------
# 4.2 Montar e treinar a rede
# ---------------------------------------------------------------
# Arquitetura pequena de proposito: caracteres binarizados de 32x32 sao um
# problema simples, rede grande so traria tempo de treino e overfitting.
aumentacao = keras.Sequential([
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08),
    layers.RandomRotation(0.02),
], name="aumentacao")

modelo_cnn = keras.Sequential([
    layers.Input(shape=(32, 32, 1)),
    layers.Rescaling(1./255),
    aumentacao,
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(2),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(CLASSES_MODELO), activation="softmax"),
], name="cnn_caracteres")

modelo_cnn.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
modelo_cnn.summary()

os.makedirs(f"{RAIZ}/modelos", exist_ok=True)
CAMINHO_CNN = f"{RAIZ}/modelos/cnn_chars.keras"
callbacks = [
    keras.callbacks.ModelCheckpoint(CAMINHO_CNN, save_best_only=True,
                                    monitor="val_accuracy"),
    keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True,
                                  monitor="val_accuracy"),
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
]

# Peso por classe (balanceado pelo inverso da frequencia no treino): sem
# isso, classes raras como Q (120 exemplos) e O (228) ficam invisiveis pra
# rede -- ver DIARIO Dia 3 (Q e O sao as classes mais raras, e visualmente
# parecidas com 0) e a matriz de confusao da primeira rodada deste notebook
# (Q e O com 0% de acerto no teste).
contagem_treino = {c: len(glob.glob(f"{DESTINO_CHARS}/train/{c}/*")) for c in CLASSES_MODELO}
n_total_treino = sum(contagem_treino.values())
peso_classe = {i: n_total_treino / (len(CLASSES_MODELO) * contagem_treino[c])
              for i, c in enumerate(CLASSES_MODELO)}
print("Peso por classe (balanceado):",
      {c: round(peso_classe[i], 2) for i, c in enumerate(CLASSES_MODELO)})

historico = modelo_cnn.fit(treino, validation_data=val, epochs=30, callbacks=callbacks,
                           class_weight=peso_classe)

perda, acc = modelo_cnn.evaluate(teste, verbose=0)
print(f"\nAcuracia no teste: {acc:.4f}")
print(f"Epocas rodadas: {len(historico.history['loss'])}")
print(f"Numero de parametros: {modelo_cnn.count_params()}")

# ---------------------------------------------------------------
# 4.3 Curvas de treino
# ---------------------------------------------------------------
os.makedirs(f"{RAIZ}/resultados/figuras", exist_ok=True)
os.makedirs(f"{RAIZ}/resultados/tabelas", exist_ok=True)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
a1.plot(historico.history["loss"], label="treino")
a1.plot(historico.history["val_loss"], label="validacao")
a1.set_title("Perda"); a1.set_xlabel("epoca"); a1.legend(); a1.grid(alpha=.3)
a2.plot(historico.history["accuracy"], label="treino")
a2.plot(historico.history["val_accuracy"], label="validacao")
a2.set_title("Acuracia"); a2.set_xlabel("epoca"); a2.legend(); a2.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{RAIZ}/resultados/figuras/curvas_cnn.png", dpi=140, bbox_inches="tight")
print("Figura de curvas salva.")

# ---------------------------------------------------------------
# 4.4 Matriz de confusao e os pares problematicos
# ---------------------------------------------------------------
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

y_real, y_prev = [], []
for lote_x, lote_y in teste:
    y_real.extend(lote_y.numpy())
    y_prev.extend(np.argmax(modelo_cnn.predict(lote_x, verbose=0), axis=1))

cm = confusion_matrix(y_real, y_prev)

plt.figure(figsize=(13, 11))
sns.heatmap(cm, xticklabels=CLASSES_MODELO, yticklabels=CLASSES_MODELO,
            cmap="Blues", square=True, cbar_kws={"shrink": .6})
plt.xlabel("previsto"); plt.ylabel("real"); plt.title("Matriz de confusao")
plt.tight_layout()
plt.savefig(f"{RAIZ}/resultados/figuras/matriz_confusao.png", dpi=140, bbox_inches="tight")
print("Figura de matriz de confusao salva.")

pares = [(CLASSES_MODELO[i], CLASSES_MODELO[j], int(cm[i, j]))
         for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
pares.sort(key=lambda t: -t[2])
print("Pares mais confundidos (real -> previsto):")
for real, prev, n in pares[:12]:
    print(f"  {real} -> {prev}: {n}")

pd.DataFrame(pares[:20], columns=["real", "previsto", "n"]).to_csv(
    f"{RAIZ}/resultados/tabelas/pares_confundidos.csv", index=False)

relatorio = classification_report(y_real, y_prev, target_names=CLASSES_MODELO,
                                  output_dict=True, zero_division=0)
pd.DataFrame(relatorio).T.to_csv(f"{RAIZ}/resultados/tabelas/classification_report_cnn.csv")

# ---------------------------------------------------------------
# Resumo final -> JSON no Drive, pra colar no DIARIO sem digitar a mao
# ---------------------------------------------------------------
resumo_final = dict(
    acuracia_teste=round(float(acc), 4),
    perda_teste=round(float(perda), 4),
    parametros=int(modelo_cnn.count_params()),
    epocas_rodadas=len(historico.history["loss"]),
    n_classes=len(CLASSES_MODELO),
    pares_mais_confundidos=[{"real": r, "previsto": p, "n": n} for r, p, n in pares[:12]],
    geracao_chars=resumo_chars,
)
with open(f"{RAIZ}/resultados/tabelas/resumo_dia4.json", "w") as f:
    json.dump(resumo_final, f, indent=2, ensure_ascii=False)
print("\nResumo do Dia 4 salvo em resultados/tabelas/resumo_dia4.json")
print(json.dumps(resumo_final, indent=2, ensure_ascii=False))
