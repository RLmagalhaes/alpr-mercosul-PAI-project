# ALPR Mercosul — Detecção e Reconhecimento de Placas Veiculares

Trabalho final da disciplina **Processamento e Análise de Imagens** (Pós-Graduação).
Sistema que recebe a foto de um veículo e devolve o texto da placa.

```
foto → [YOLO detecta a placa] → [recorte com margem de 8%]
     → [YOLO detecta e classifica os 7 caracteres]
     → [regra do formato brasileiro] → "ABC1D23"
```

> A primeira versão usava segmentação clássica (CLAHE + Otsu + componentes
> conectados) para separar os caracteres. Ela continua no código, em
> `src/preprocessamento.py`, porque a **comparação entre os dois métodos é o
> resultado central do trabalho** — ver a tabela abaixo.

## Resultados

Todos os números saíram de código executado; as tabelas de origem estão em
`resultados/tabelas/`.

| Métrica | Meta | Obtido |
| --- | --- | --- |
| mAP@0.5 — detecção da placa (257 imagens de teste) | > 0,90 | **0,992** ✅ |
| mAP@0.5:0.95 — detecção da placa | — | 0,834 |
| Precisão / recall — detecção da placa | — | 0,978 / 0,977 |
| mAP@0.5 — detecção de caracteres (36 imagens de teste) | — | 0,931 |
| Acurácia por caractere — CNN isolada, recortes exatos | > 0,95 | 0,944 ⚠️ |
| **Acurácia por caractere — fim a fim, 30 placas BR** | > 0,95 | **0,719** ❌ |
| **Acurácia por placa — fim a fim, 30 placas BR** | > 0,80 | **0,333** ❌ |

**Duas das três metas não foram atingidas.** O sistema lê corretamente 10 das 30
placas do conjunto de avaliação. O relatório discute por quê, sem maquiar.

### O resultado central: o gargalo era a segmentação

As mesmas 30 placas brasileiras, com gabarito conferido por humano
(`resultados/tabelas/comparacao_pipelines.csv`):

| Pipeline | Acurácia por caractere | Acurácia por placa |
| --- | --- | --- |
| A) segmentação clássica + CNN | 0,300 | 0/30 |
| B) YOLO só para as **caixas** + a **mesma** CNN | 0,619 | 7/30 |
| **C) YOLO caixa + classe** | **0,719** | **10/30** |

O pipeline **B** é a prova: mantendo a CNN idêntica dos dois lados e trocando
**apenas** a forma de localizar os caracteres, a acurácia por caractere dobra e a
acurácia por placa sai de zero. O gargalo não era a CNN, nem o dataset, nem o
detector de placas — era a segmentação.

## Como reproduzir

```bash
git clone https://github.com/RLmagalhaes/alpr-mercosul-PAI-project.git
cd alpr-mercosul-PAI-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v                     # 28 testes das funções puras
```

Inferência e avaliação rodam em **CPU local**, em segundos. Só o treino precisou
de GPU (T4 do Google Colab).

Os modelos treinados (`.pt`, `.keras`) e os datasets **não estão versionados**,
por tamanho — ver `.gitignore`. Os três modelos usados pelo sistema estão na
pasta da entrega (`entrega/modelos/`, 29 MB), montada por
`notebooks/11_montar_entrega.py`.

### Chave do Roboflow

Os datasets vêm do Roboflow Universe, e os notebooks 01, 03, 04, 05 e 06 baixam
os dados por API. **A chave não fica no código.** Forneça a sua assim:

```bash
export ROBOFLOW_API_KEY="sua-chave"          # localmente
# ou, na VM do Colab:
echo 'open("/content/.roboflow_key","w").write("SUA_CHAVE")' | colab exec
```

> ⚠️ Até o Dia 7 havia uma chave escrita no código dos notebooks. Ela **foi
> revogada** e continua visível no histórico do Git. Se alguém a recuperar de um
> commit antigo e tentar usá-la, vai receber
> `401 This API key does not exist or has been revoked` — isso é esperado, não é
> bug. Gere uma chave nova em app.roboflow.com → Settings → API Keys.

## Estrutura

```
├── RELATORIO.md / .html     relatório completo (fonte e versão para impressão)
├── DIARIO.md                registro dia a dia, com as métricas de cada etapa
├── docs/
│   ├── ROTEIRO_ALPR_7_DIAS.md   o plano original
│   ├── RETOMAR.md               onde o projeto parou e como continuar
│   └── COLAB_SKILL.md           referência da Colab CLI
├── notebooks/               01 a 11, na ordem de execução
├── src/                     módulos do sistema — a fonte da verdade
│   ├── preprocessamento.py      recortar, endireitar, detectar_layout, segmentar
│   ├── validacao.py             regra do formato Mercosul / antigo
│   ├── pipeline.py              LeitorDePlacas (rota clássica)
│   ├── leitor_yolo.py           LeitorDePlacasYOLO (rota final)
│   └── metricas.py              iou, acuracia_caractere, acuracia_placa
├── tests/                   28 testes das funções puras (pytest, sem GPU)
├── api/                     esqueleto de API REST em FastAPI — escrito, porém
│                            não avaliado; ainda aponta para o pipeline clássico
└── resultados/
    ├── figuras/             as 13 figuras citadas no relatório
    └── tabelas/             todos os números, em CSV
```

## Dados

- **Detecção:** `trafficbr/vehicle-plate-color` v2 (Roboflow Universe) — 12.780 /
  960 / 257 imagens de veículos com a placa anotada, placas brasileiras.
- **Caracteres:** `project-swcsj/license-plate-character-extraction` v2 — 36
  classes, ~33 mil caracteres anotados individualmente. Usado de duas formas:
  como fonte de recortes para a CNN e como dataset de detecção para o YOLO de
  caracteres.
- **Conjunto de avaliação:** 30 placas brasileiras (16 Mercosul, 14 formato
  antigo) com o texto transcrito e conferido por humano, em
  `resultados/tabelas/gabarito_mercosul.csv`.
- **Upgrade natural:** [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset)
  — 20 mil imagens brasileiras já transcritas, o que eliminaria o gabarito manual.

## Onde o projeto parou

O pipeline de visão está completo e medido. O que ficou em aberto — fechar as 40
épocas do detector de caracteres, recalibrar o limiar de rejeição, ampliar o
gabarito e ligar a API ao pipeline final — está descrito, com o contexto
necessário para retomar, em **[`docs/RETOMAR.md`](docs/RETOMAR.md)**.

## Privacidade

Placa veicular é dado que pode identificar indiretamente uma pessoa. Este projeto
é um exercício acadêmico, sem finalidade de vigilância ou rastreamento. Foram
usados apenas conjuntos públicos licenciados para pesquisa; nenhuma imagem foi
coletada pelo autor; e o sistema produz apenas a string da placa — não consulta,
armazena nem cruza qualquer dado sobre proprietário, veículo ou localização.

Um sistema com esta acurácia **não deve ser usado para decisões automáticas sobre
pessoas**. O mecanismo de recusa implementado (devolver `revisao_manual` em vez de
arriscar um palpite) é, antes de escolha de engenharia, a postura correta para um
sistema cujos erros recaem sobre terceiros.

## Licença

Uso acadêmico.
