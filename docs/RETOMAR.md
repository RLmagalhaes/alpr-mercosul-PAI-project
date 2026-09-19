# Como retomar este projeto

> Escrito no Dia 7, quando o trabalho foi entregue à professora e o projeto
> entrou em pausa. O objetivo é que, meses depois, dê para voltar sem reler o
> `DIARIO.md` inteiro (47 KB) nem redescobrir as armadilhas na marra.
>
> Leitura na ordem: **esta página** → o `RELATORIO.md` (o que o sistema faz e
> quanto acerta) → o `DIARIO.md` só se precisar do detalhe de um dia específico.

---

## 1. Em uma página: onde o projeto parou

O pipeline de visão computacional está **completo, medido e documentado**:

```
foto → YOLO detecta a placa (mAP@0.5 = 0,992)
     → recorte com margem de 8%
     → YOLO detecta e classifica os 7 caracteres (mAP@0.5 = 0,931)
     → regra do formato brasileiro (+10 pontos de acurácia)
     → "ABC1D23"   — acerta 10 de 30 placas (0,333)
```

**Duas metas não foram atingidas:** acurácia por caractere 0,719 (meta 0,95) e
por placa 0,333 (meta 0,80). Isso está declarado no relatório, não escondido.

O que o trabalho entrega no lugar é um **diagnóstico medido**: trocando só a
forma de localizar os caracteres e mantendo a mesma CNN, a acurácia por
caractere dobra. O gargalo era a segmentação clássica, não a rede.

**O que foi cortado do escopo por falta de tempo**, e continua no repositório
sem uso: ONNX, API em produção, Docker, medição de latência e os 10 slides.
O `api/app.py` existe e está escrito, mas nunca foi executado nem medido — e
ainda aponta para `LeitorDePlacas` (a rota clássica, 0,300), não para
`LeitorDePlacasYOLO` (a rota final, 0,719).

---

## 2. Voltar a rodar — 5 minutos

```bash
cd "alpr-mercosul"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v          # esperado: 28 passed
```

**Tudo que não é treino roda em CPU local, em segundos.** O `.venv` do Mac tem
TensorFlow 2.17 e OpenCV 4.10; a avaliação completa das 30 placas leva menos de
um minuto. Colab só é necessário para treinar.

Teste rápido de que o sistema está vivo:

```python
from src.leitor_yolo import LeitorDePlacasYOLO

leitor = LeitorDePlacasYOLO("modelos/detector_best.pt",
                            "modelos/chars_best.pt",
                            "modelos/cnn_chars_compat.keras")
print(leitor.ler("entrega/imagens_exemplo/<alguma>.jpg", modo="yolo"))
```

**Onde estão os artefatos pesados** (todos fora do Git, por tamanho):

| O quê | Onde | Tamanho |
| --- | --- | --- |
| Os 3 modelos usados pelo sistema | `modelos/` e `entrega/modelos/` | 29 MB |
| Datasets baixados do Roboflow | `dados/` | 941 MB |
| Pasta montada para a professora | `entrega/` | 42 MB |

Se `dados/` tiver sumido, os notebooks 01 e 03 rebaixam tudo do Roboflow — mas
é preciso uma chave nova (ver §5).

> **Por que o clone baixa ~50 MB se o repositório tem 12 MB:** um zip de 38 MB da
> entrega entrou por engano no commit `541efca` e foi removido no commit seguinte.
> O blob continua no histórico — decisão consciente de não reescrever o histórico
> publicado. Não afeta o conteúdo, só o tamanho do clone.

---

## 3. O mapa do código

Os módulos em `src/` são a fonte da verdade; os notebooks importam deles.

| Arquivo | O que faz |
| --- | --- |
| `src/leitor_yolo.py` | **`LeitorDePlacasYOLO` — a rota final.** `ler(img, modo="yolo")` |
| `src/pipeline.py` | `LeitorDePlacas` — a rota clássica, mantida para a comparação |
| `src/preprocessamento.py` | `recortar`, `endireitar`, `detectar_layout`, `segmentar` |
| `src/validacao.py` | `aplicar_mascara` — a regra do formato, 15 linhas |
| `src/metricas.py` | `iou`, `acuracia_caractere`, `acuracia_placa`, `erros_por_posicao` |

Notebooks, na ordem em que foram executados:

| # | O que faz |
| --- | --- |
| 01 | Baixa o dataset de detecção e treina o YOLO de placas |
| 02 | Avalia a detecção (mAP, IoU, piores casos) |
| 03 | Pré-processamento e geração do dataset de recortes de caracteres |
| 04 | Treina a CNN de 36 classes |
| 05 | Pipeline clássico fim a fim · `05b` mede a ablação da segmentação |
| 06 | `gabarito_mercosul` monta as 30 placas · os `yolo_caracteres_*` preparam/treinam/acompanham o detector de caracteres |
| 07 | Compara os 3 pipelines sobre as mesmas 30 placas — **o experimento central** |
| 08 | Varredura do limiar de confiança e erros por posição |
| 09 | Avalia o YOLO de caracteres isoladamente |
| 10 | Gera `RELATORIO.html` (completo) e `RELATORIO_entrega.html` (sem 6.4/6.5) |
| 11 | Monta a pasta `entrega/` |

Dois constantes que valem conhecer: `CONF_CARACTERE = 0,05` e
`LIMIAR_CONFIANCA = 0,70`, ambas em `src/leitor_yolo.py`.

---

## 4. As pendências, em ordem de retorno por esforço

### 4.1 Fechar as 40 épocas do detector de caracteres — *maior retorno, menor esforço*

O `modelos/chars_best.pt` em uso é o checkpoint da **época ~30 de 40**: a sessão
do Colab caiu e o treino nunca foi retomado. **Há margem de melhora não medida**,
e é a mudança mais barata do projeto.

- Retomar: `notebooks/06_yolo_caracteres_treinar.py`, com `resume=True`
  apontando para o `last.pt` salvo no Drive.
- Reavaliar com `09_avaliar_yolo_caracteres.py` (mAP) e
  `07_comparar_pipelines.py` (acurácia fim a fim nas 30 placas).
- **Como saber se melhorou:** a linha `C) YOLO caixa+classe — com regra` em
  `resultados/tabelas/comparacao_pipelines.csv` hoje marca `0.719 / 0.3333`.

### 4.2 Recalibrar o limiar de rejeição

`LIMIAR_CONFIANCA = 0,70` nunca foi ajustado com dados reais. A confiança mínima
observada fica entre **0,059 e 0,759, mediana 0,344** — ou seja, o sistema marca
quase tudo como `revisao_manual`. Conservador e seguro, mas pouco útil.

- Os dados da varredura já existem:
  `resultados/tabelas/varredura_conf_caracteres.csv`.
- O certo é escolher o limiar por curva de precisão×cobertura sobre o gabarito,
  não por palpite.

### 4.3 Ampliar o gabarito de 30 para 100–200 placas

Com 30 placas, **cada uma vale 3,3 pontos** de acurácia — não dá para separar
0,33 de 0,45 com confiança. `notebooks/06_gabarito_mercosul.py` já monta o
conjunto; é trabalho braçal de transcrição.

> ⚠️ **Armadilha metodológica já encontrada:** não use a regra de formato para
> desambiguar `O` × `0` ao transcrever. O sistema avaliado também usa essa regra,
> e o gabarito ficaria circular, inflando a acurácia. No Dia 6 as 8 leituras
> duvidosas foram conferidas a olho, uma a uma.

### 4.4 Corrigir a perspectiva

Hoje só a **rotação no plano** é corrigida (`endireitar`). Retificação por
homografia a partir dos 4 cantos da placa resolveria a perspectiva — provável
ganho relevante, já que o detector de caracteres sofre com placas em ângulo.

### 4.5 Melhorar a detecção de layout

`detectar_layout` acerta **24 de 30**. Cada erro aplica a máscara errada e
corrompe a correção justamente onde ela deveria ajudar. Hoje é um score de azul
na tarja superior; um classificador binário sobre o recorte seria mais robusto.

### 4.6 Ligar a API ao pipeline final

`api/app.py` instancia `LeitorDePlacas` (clássico, 0,300). Trocar por
`LeitorDePlacasYOLO` (0,719) é uma linha. Depois disso é que faz sentido medir
latência, exportar para ONNX e subir o Docker — a sequência do Dia 6 original,
que está descrita em `docs/ROTEIRO_ALPR_7_DIAS.md`.

### 4.7 Trocar de base

[RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset): 20 mil
imagens brasileiras **já transcritas**. Elimina o gabarito manual e permite
avaliar em escala real. Exige solicitação por e-mail, liberada em 1 a 5 dias
úteis — foi recusada no Dia 0 por falta de tempo, mas numa retomada sem prazo é
a melhor decisão disponível.

### 4.8 Fora do escopo atual

Placas de motocicleta (texto em duas linhas) e condições noturnas — as bases
usadas são majoritariamente diurnas e de carros.

---

## 5. Armadilhas conhecidas

Cada uma destas custou tempo real. Estão no `DIARIO.md`, seção "Registro de
problemas", com mais detalhe.

**Colab / infraestrutura**

- `colab exec` tem timeout padrão de **30 s**. Passe `--timeout 600` em scripts
  longos, sempre.
- `colab upload` é instável (falha com 500 sem motivo). Contorno que funciona:
  mandar o conteúdo do arquivo por stdin, com `open(caminho,'w').write(...)`.
- `colab drivemount` **exige autenticação interativa** — precisa ser rodado por
  você no terminal. Disparado por um agente dá `ValueError: mount failed`, e
  abrir sessão nova não resolve.
- O kernel **mantém em memória os módulos já importados**. Depois de atualizar
  qualquer coisa em `src/` na VM, rode `colab restart-kernel` — sem isso você
  fica testando código velho, sem nenhum erro avisando.
- Sessão gratuita cai sem aviso. A estratégia que salvou o projeto duas vezes:
  copiar o `last.pt` para o Drive a cada 3 minutos e usar `resume=True`. Quando
  a sessão caiu na época 30, **nenhuma época precisou ser refeita** — só o
  websocket havia caído, a VM seguiu treinando.

**Modelos e dados**

- O `.keras` salvo pela VM (Keras 3.13) **não abre no Keras local** (3.10, porque
  o venv é Python 3.9). Use `modelos/cnn_chars_compat.keras` — é o mesmo modelo,
  com a chave `quantization_config` removida do `config.json` interno.
- O dataset `trafficbr` tem **anotações grosseiramente erradas**: caixas
  rotuladas `plate` que são carros inteiros (45-63% da imagem). Ordenar por área
  sem filtrar seleciona justamente esses erros. O filtro que funciona: razão
  largura/altura entre 1,8 e 5,0, e no máximo 25% da imagem.
- Antes de qualquer treino longo, faça um **ensaio de 3 épocas**. Erro de
  caminho aparece em 2 minutos em vez de 40.

**Chave do Roboflow**

A chave que ficava escrita nos notebooks **foi revogada** e continua visível no
histórico do Git (commits `2dfa670`, `b2f6457`, `f03f9b5`, `235e06d`). Se você
recuperá-la de um commit antigo, vai receber
`401 This API key does not exist or has been revoked` — **é esperado, não é
bug**. Gere uma chave nova em app.roboflow.com → Settings → API Keys e
disponibilize por `ROBOFLOW_API_KEY` ou por `/content/.roboflow_key` na VM.
O histórico não foi reescrito de propósito: a chave está inativa, e reescrever
21 commits quebraria o remote sem ganho de segurança.

---

## 6. Onde está cada coisa

| Documento | Para quê |
| --- | --- |
| `RELATORIO.md` / `RELATORIO.html` | Versão completa, com 6.4 (trabalhos futuros) e 6.5 (privacidade) |
| `RELATORIO_entrega.html` | Versão entregue à professora, sem 6.4 e 6.5 — gerada por `notebooks/10_*.py` |
| `entrega/` | A pasta montada para a professora, com PDF, notebook, código, modelos e resultados |
| `DIARIO.md` | Registro dia a dia, com as métricas de cada etapa e o histórico de erros |
| `docs/ROTEIRO_ALPR_7_DIAS.md` | O plano original de 7 dias, incluindo o Dia 6 (produção) que foi cortado |
| `docs/COLAB_SKILL.md` | Referência completa da Colab CLI |
| `CLAUDE.md` | Instruções para o agente de IA que trabalhou no projeto |

Os planos de trabalho gerados durante o projeto ficaram **fora do repositório**,
em `~/.claude/plans/` (`eu-j-estou-atrasado-*.md` e
`pensando-no-resultado-final-*.md`).

---

## 7. Se for retomar do zero, comece por aqui

1. `pytest tests/ -v` — se os 28 passarem, o código está íntegro.
2. Rode `notebooks/07_comparar_pipelines.py` e confirme que reproduz
   `0.719 / 0.3333`. Se reproduzir, o sistema inteiro está funcionando.
3. Ataque a §4.1 (fechar as 40 épocas). É a única pendência com ganho provável
   sem trabalho novo de modelagem.
4. Só depois pense em trocar de base (§4.7) — aí vira outro projeto, maior e
   melhor.

---

## 8. Decisões de limpeza em aberto

Uma varredura de redundâncias foi feita no fim do Dia 7. O que era consensual já
foi aplicado (ver `DIARIO.md`). **Estes sete itens ficaram para decisão**, porque
cada um tem argumento dos dois lados. Revisitar na próxima sessão.

### 8.1 O `api/` fica ou sai? — *decidir primeiro, os outros dependem*

79 linhas (`app.py`, `Dockerfile`, `requirements.txt`) escritas, **nunca
executadas**, e `app.py` instancia `LeitorDePlacas` (acurácia 0,300) em vez de
`LeitorDePlacasYOLO` (0,719) — está errado, não só parado.

- **Fica:** é a continuação natural do projeto (§4.6) e trocar a classe é uma
  linha. Nesse caso `fastapi`, `uvicorn` e `python-multipart` seguem no
  `requirements.txt`.
- **Sai:** o `requirements.txt` cai de 12 para 9 linhas e o repositório deixa de
  ter uma pasta que promete algo que não funciona.

### 8.2 As 9,5 MB de figuras que o relatório não exibe

`resultados/figuras/` tem 13 PNGs e **nenhum aparece no relatório** — são zero
`![...]` no Markdown e zero `<img>` no HTML. O PDF entregue é texto e tabelas.

- **Ficam:** são a trilha de evidência de que cada número saiu de código.
- **Encolhem:** as 4 maiores somam 7,1 MB (`gabarito_para_rotular` 2,5 MB,
  `pipeline_exemplos` 1,7 MB, `piores_casos` 1,5 MB, `verificacao_recorte_dia5`
  1,4 MB) e as duas últimas são diagnóstico interno, não resultado.
- **Recomprimem:** `dpi=140` em PNG é pesado para o que elas mostram.

Para dimensionar: hoje **63% do repositório versionado são esses 13 arquivos.**

### 8.3 Renumerar os notebooks

A ordem de execução não se lê na lista de arquivos: há **quatro** notebooks
`06_`, um `05b`, e dois sem número nenhum — sendo que `treinar_detector.py` é o
treino do Dia 1, o passo mais importante da sequência, e fica solto no fim da
ordem alfabética. Renumerar de 01 a 14 resolveria; o custo é quebrar os caminhos
citados no `RELATORIO.md`, no `README.md` e neste arquivo.

### 8.4 O notebook da professora mora dentro de um script

`notebooks/11_montar_entrega.py` tem 626 linhas, das quais **372 (59%) são o
`ALPR_Mercosul.ipynb` escrito como strings Python**. Como `entrega/` é ignorada
pelo Git, o notebook entregue não está versionado em lugar nenhum — só a receita
para gerá-lo. Versionar o `.ipynb` e fazer o script apenas copiá-lo seria mais
simples de entender, ao custo de manter os dois em sincronia.

### 8.5 A duplicação entre `pipeline.py` e `leitor_yolo.py`

**33 das 115 linhas** de `pipeline.py` são idênticas às de `leitor_yolo.py` — o
carregamento preguiçoso dos modelos e a leitura da imagem. Uma classe base
resolveria em ~30 minutos. (A constante `LIMIAR_CONFIANCA`, que estava definida
duas vezes, já foi unificada em `validacao.py`.)

A duplicação equivalente entre `treinar_detector.py` e
`06_yolo_caracteres_treinar.py` (o `sync_loop` de checkpoint) **não vale
corrigir**: esses scripts rodam isolados dentro da VM do Colab, onde `src/` pode
não existir.

### 8.6 Os HTML gerados estão versionados

`RELATORIO.html` (29 KB) e `RELATORIO_entrega.html` (27 KB) são derivados de
`RELATORIO.md` e regeneram em 1 segundo. Versionar saída de build normalmente é
errado. O `RELATORIO_entrega.pdf` (423 KB) é caso diferente: é o artefato que foi
efetivamente entregue, e vale manter por rastreabilidade.

> As ~22 linhas que geram a versão sem as seções 6.4 e 6.5
> (`SECOES_FORA_DA_ENTREGA` em `notebooks/10_gerar_html_relatorio.py`)
> **devem ficar**. Se o relatório for reimpresso sem elas, as duas seções voltam
> silenciosamente para o PDF da professora.

### 8.7 Risco, não redundância: os modelos só existem num lugar

Os três modelos treinados (29 MB) estão **apenas neste Mac**. Não vão para o
GitHub, corretamente, por tamanho — mas o README não tem link de backup. Se o
disco falhar, são 40 épocas de treino perdidas. **Subir para o Drive e colar o
link no README é a tarefa mais barata e mais valiosa desta lista.**
