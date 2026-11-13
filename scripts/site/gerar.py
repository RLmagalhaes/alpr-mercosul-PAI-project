"""Monta a página de estudo de caso (`site/index.html`) a partir dos CSVs.

Todo número da página sai daqui, lido de `resultados/tabelas/`. Não roda modelo:
as imagens da galeria vêm de `scripts/site/preparar_galeria.py`.

Uso:
    .venv/bin/python scripts/site/gerar.py
    # depois, para ver no navegador:
    python -m http.server -d site 8000      # e abra http://localhost:8000
"""

import os
import shutil
import sys

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)

from src.validacao import LIMIAR_CONFIANCA, MASCARAS  # noqa: E402

TAB = f"{RAIZ}/resultados/tabelas"
MODELO = f"{RAIZ}/scripts/site/modelo"
SITE = f"{RAIZ}/site"

REPO = "https://github.com/RLmagalhaes/alpr-mercosul-PAI-project"
N_CARACTERES = 7

# Placa usada no topo da página e na explicação das etapas.
N_DESTAQUE = 7
N_ETAPAS = 2

# O que está em andamento. Os números de issue são do GitHub; o texto é nosso.
PROXIMOS_PASSOS = [
    (6, "Ampliar o gabarito de 30 para 100 a 200 placas, para que cada placa pese "
        "menos de 1 ponto na acurácia."),
    (4, "Recalibrar o limiar de revisão manual, para liberar mais acertos sem "
        "deixar passar erros."),
    (3, "Corrigir a perspectiva da placa antes de ler os caracteres."),
    (5, "Tornar mais robusta a detecção do formato (Mercosul ou antigo)."),
    (2, "Ler placas de moto, que têm o texto em duas linhas."),
    (9, "Deixar o visitante testar o modelo com a própria foto, no navegador."),
]


# ---------- formatação em português ----------

def num(valor: float, casas: int = 3) -> str:
    """0.719 -> '0,719'."""
    return f"{valor:.{casas}f}".replace(".", ",")


def pct(valor: float, casas: int = 0) -> str:
    """0.977 -> '98%'."""
    return f"{valor * 100:.{casas}f}".replace(".", ",") + "%"


# ---------- leitura dos CSVs ----------

def ler_deteccao():
    d = pd.read_csv(f"{TAB}/metricas_deteccao.csv").iloc[0]
    inv = pd.read_csv(f"{TAB}/inventario.csv")
    chars = pd.read_csv(f"{TAB}/metricas_yolo_caracteres.csv")
    chars = chars[chars["split"] == "test"].iloc[0]
    return dict(
        map50=num(d["mAP@0.5"]), recall=pct(d["Recall"]),
        precisao=pct(d["Precisao"]),
        n_teste=int((inv["split"] == "test").sum()),
        n_treino=f"{int((inv['split'] == 'train').sum()):,}".replace(",", "."),
        chars_map50=num(chars["mAP50"]), chars_n_teste=int(chars["n_imagens"]),
    )


def ler_rotas():
    """As três rotas, sempre com a regra do formato aplicada."""
    df = pd.read_csv(f"{TAB}/comparacao_pipelines.csv")
    df = df[df["pipeline"].str.contains("com regra")]
    nomes = {"A": "Segmentação clássica + CNN",
             "B": "YOLO acha as caixas + a mesma CNN",
             "C": "YOLO acha e classifica (rota final)"}
    rotas = []
    for _, r in df.iterrows():
        letra = r["pipeline"][0]
        rotas.append(dict(nome=nomes[letra], final=(letra == "C"),
                          car=r["acc_caractere"], car_txt=num(r["acc_caractere"]),
                          placa=r["acc_placa"], placa_txt=num(r["acc_placa"])))
    return rotas


def ler_leitor():
    """Acurácia final e o que o limiar de revisão faz com acertos e erros."""
    det = pd.read_csv(f"{TAB}/leitor_final_detalhe.csv", dtype=str).fillna("")
    det["conf_minima"] = det["conf_minima"].astype(float)
    det["n_caracteres"] = det["n_caracteres"].astype(int)
    det["acertou"] = det["real"] == det["com_regra"]
    # mesma regra de src/leitor_yolo.py: 7 caracteres e confiança >= limiar
    det["liberada"] = ((det["n_caracteres"] == N_CARACTERES) &
                       (det["conf_minima"] >= LIMIAR_CONFIANCA))

    final = pd.read_csv(f"{TAB}/leitor_final.csv")
    final = final[final["regra"] == "com regra"].iloc[0]
    n = len(det)
    acertos = int(det["acertou"].sum())
    return dict(
        n_placas=n, acertos=acertos, erros=n - acertos,
        acc_car=num(final["acc_caractere"]), acc_placa=num(final["acc_placa"]),
        acc_car_pct=pct(final["acc_caractere"]),
        peso_placa=num(100 / n, 1),
        limiar=num(LIMIAR_CONFIANCA, 2),
        acertos_liberados=int((det["acertou"] & det["liberada"]).sum()),
        erros_liberados=int((~det["acertou"] & det["liberada"]).sum()),
    )


def ler_erros_posicao():
    df = pd.read_csv(f"{TAB}/erros_por_posicao.csv")
    sem = df[df["regra"] == "sem regra"].iloc[0]
    com = df[df["regra"] == "com regra"].iloc[0]
    colunas = [f"pos{i + 1}" for i in range(N_CARACTERES)]
    maximo = max(int(com[c]) for c in colunas)
    posicoes = [dict(pos=i + 1, com=int(com[c]), sem=int(sem[c]),
                     mercosul=MASCARAS["mercosul"][i], antiga=MASCARAS["antiga"][i],
                     altura=int(com[c]) / maximo, pior=(int(com[c]) == maximo))
                for i, c in enumerate(colunas)]
    piores = [str(p["pos"]) for p in posicoes if p["pior"]]
    return dict(posicoes=posicoes, piores=" e ".join(piores),
                total_sem=int(sum(sem[c] for c in colunas)),
                total_com=int(sum(com[c] for c in colunas)))


def placa(texto: str, real: str = "") -> list:
    """Quebra a leitura em 7 casas e marca as que diferem do gabarito."""
    casas = []
    for i in range(N_CARACTERES):
        c = texto[i] if i < len(texto) else ""
        errada = bool(real) and (c != (real[i] if i < len(real) else ""))
        casas.append(dict(c=c, errada=errada))
    return casas


def ler_galeria():
    df = pd.read_csv(f"{TAB}/galeria.csv", dtype=str).fillna("")
    exemplos = []
    for _, r in df.iterrows():
        conf = float(r["conf_minima"])
        n_car = int(r["n_caracteres"])
        exemplos.append(dict(
            n=int(r["n"]), real=r["real"], lido=r["com_regra"],
            sem_regra=r["sem_regra"], layout=r["layout"], motivo=r["motivo"],
            acertou=(r["real"] == r["com_regra"]),
            # a regra só merece menção quando foi ela que fez a leitura acertar
            regra_corrigiu=(r["sem_regra"] != r["com_regra"]
                            and r["real"] == r["com_regra"]),
            liberada=(r["status"] == "ok"),
            conf=num(conf, 2), n_caracteres=n_car,
            casas=placa(r["com_regra"], r["real"]),
            foto=f"img/galeria/{r['n']}_foto.jpg",
            recorte=f"img/galeria/{r['n']}_recorte.png",
        ))
    return exemplos


def main():
    galeria = ler_galeria()
    por_n = {e["n"]: e for e in galeria}
    contexto = dict(
        repo=REPO, det=ler_deteccao(), rotas=ler_rotas(), leitor=ler_leitor(),
        posicoes=ler_erros_posicao(), galeria=galeria,
        destaque=por_n[N_DESTAQUE], etapas=por_n[N_ETAPAS],
        proximos=PROXIMOS_PASSOS,
        n_acertos_galeria=sum(e["acertou"] for e in galeria),
        n_erros_galeria=sum(not e["acertou"] for e in galeria),
    )

    env = Environment(loader=FileSystemLoader(MODELO),
                      undefined=StrictUndefined, autoescape=True)
    html = env.get_template("index.html").render(**contexto)

    os.makedirs(SITE, exist_ok=True)
    with open(f"{SITE}/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    shutil.copy(f"{MODELO}/estilo.css", f"{SITE}/estilo.css")
    print(f"-> {SITE}/index.html")
    print(f"   {contexto['leitor']['acertos']}/{contexto['leitor']['n_placas']} "
          f"placas certas, detecção mAP@0.5 {contexto['det']['map50']}")


if __name__ == "__main__":
    main()
