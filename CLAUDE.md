# CLAUDE.md — Instruções do projeto

> Este arquivo é lido automaticamente pelo Claude Code em toda sessão.
> Não é preciso recolar o briefing a cada dia.

---

## O projeto

Sistema **ALPR Mercosul**: recebe a foto de um veículo e devolve o texto da placa.

```
foto → [YOLO detecta a placa] → [recorte + endireitamento] → [cinza + CLAHE + Otsu]
     → [segmenta 7 caracteres] → [CNN classifica cada um] → [regra do formato] → "ABC1D23"
```

Trabalho final de pós-graduação em Processamento e Análise de Imagens.
**Prazo: 7 dias, 2 a 3 horas por dia.** O plano completo está em `docs/ROTEIRO_ALPR_7_DIAS.md`.

**Metas:** mAP@0.5 > 0,90 na detecção · acurácia por caractere > 0,95 · acurácia por placa (os 7 corretos) > 0,80.

---

## Sobre o Raphael

- Primeiro projeto de visão computacional; concluiu um plano de estudos de 30 dias sobre redes neurais, CNNs, transfer learning, YOLO, ViT, CLIP e deploy.
- **Python intermediário:** lê e valida código com tranquilidade, mas não quer escrever do zero.
- O objetivo é entregar o projeto **e** entender o que foi feito.

---

## Como trabalhar

1. **Você escreve todo o código.** Sempre completo, comentado em português, pronto para rodar. Nunca peça para ele escrever ou completar trechos.

2. **Antes do código:** 2 a 3 linhas dizendo o que o bloco faz e por quê.

3. **Depois do código:** diga qual saída é esperada, para ele conferir.

4. **Ao ver a saída real:** explique o que aconteceu em linguagem simples, diga se o resultado está bom ou ruim, e sugira correções quando for o caso.

5. **Pergunte antes de avançar.** Ao fim de cada etapa: *"seguimos para a etapa X ou quer ajustar algo aqui?"*. Espere a resposta.

6. **Se der erro:** corrija direto, entregando o bloco corrigido inteiro, e explique em 2 linhas qual era o problema. Não faça ele adivinhar.

7. **Seja honesto com métricas.** Se ficou ruim, diga que ficou ruim e proponha o que tentar. Não maquie resultado, não elogie número fraco.

8. **Priorize ter algo funcionando.** Entre elegante e pronto, escolha pronto. Refinamento é bônus.

---

## Fluxo de cada sessão

**Ao iniciar** (o Raphael vai dizer "vamos para o Dia N"):

1. Leia `DIARIO.md` para saber onde o projeto parou.
2. Leia a seção do Dia N em `docs/ROTEIRO_ALPR_7_DIAS.md`.
3. Diga em 3 linhas: o objetivo do dia e o entregável esperado.
4. Comece a primeira etapa.

**Ao encerrar** — obrigatório, é isto que dá continuidade ao projeto:

1. Atualize `DIARIO.md` com: o que foi feito, **as métricas obtidas com os números reais**, decisões tomadas, o que ficou pendente e qual o próximo passo.
2. Faça `git add -A && git commit` com uma mensagem descritiva.
3. Resuma a sessão em 5 linhas.

---

## Ambiente de execução

O treino roda em **GPU do Google Colab**, acessada pela **Colab CLI** — não há GPU local (MacBook Air, Apple Silicon). A CLI foi instalada via `uv tool install google-colab-cli` e vive fora do venv do projeto.

**A referência completa da CLI está em `docs/COLAB_SKILL.md`** (gerada por `colab skill`). Consulte esse arquivo antes de usar qualquer comando que não esteja na lista abaixo.

Comandos verificados neste ambiente:

```bash
colab new                # cria a sessão (provisiona a VM)
colab sessions           # lista sessões ativas
colab status             # estado da sessão atual
colab drivemount         # monta o Google Drive na VM
colab install ultralytics tensorflow    # instala pacotes na VM
colab upload arquivo.py  # envia arquivo local para a VM
colab ls                 # lista arquivos na VM
colab download modelos/  # traz artefatos de volta para o Mac
colab log                # histórico da sessão
colab restart-kernel     # reinicia o kernel sem perder a VM
colab stop               # encerra a sessão — SEMPRE ao fim do dia
colab version            # versão da CLI (não existe --version)
```

**Executar código — duas formas:**

```bash
echo "print('oi')" | colab exec           # código pela entrada padrão (stdin)
colab exec -f notebooks/01_dados.ipynb    # arquivo .py ou .ipynb
colab run script.py                       # roda em VM nova e libera ao terminar
```

`colab exec` **não aceita código como argumento posicional** — ou vem por stdin, ou por `-f`.

**Regras de uso nesta máquina:**

- Sempre `colab stop` ao encerrar a sessão do dia: runtime parado não consome cota.
- Antes de um treino longo, `colab status` para confirmar que a VM está viva.
- Prefira `colab exec -f` com notebooks versionados no repositório, em vez de código solto por stdin — assim o que rodou fica registrado no Git.

**Se a Colab CLI não funcionar na conta gratuita:** o plano B é o Colab no navegador. Nesse caso, entregue cada bloco de código pronto para colar numa célula, e o Raphael cola a saída de volta para você diagnosticar. O fluxo do roteiro já prevê isso.

**Cuidados com o Colab gratuito:**
- Cota diária de GPU e sessão que cai por inatividade — evite treinos maiores que ~40 minutos seguidos.
- Salve checkpoints no Drive. Se um treino do YOLO cair, retome com `YOLO('.../weights/last.pt')` e `resume=True`.
- Antes do treino longo, faça sempre um ensaio de 3 épocas: erros de caminho aparecem em 2 minutos em vez de 40.

---

## Estrutura do repositório

```
alpr-mercosul/
├── CLAUDE.md              # este arquivo
├── DIARIO.md              # estado do projeto — atualizar todo dia
├── README.md
├── docs/
│   └── ROTEIRO_ALPR_7_DIAS.md
├── notebooks/             # 01_ a 06_, numerados na ordem de execução
├── src/
│   ├── metricas.py        # iou, acuracia_caractere, acuracia_placa
│   ├── validacao.py       # regra do formato Mercosul
│   ├── preprocessamento.py# endireitar, preparar, detectar_layout, segmentar
│   └── pipeline.py        # LeitorDePlacas.ler()
├── api/
│   ├── app.py             # FastAPI /read_plate
│   ├── requirements.txt
│   └── Dockerfile
├── tests/
│   └── test_basico.py     # roda com pytest, sem dependências pesadas
└── resultados/
    ├── figuras/           # PNGs para o relatório
    └── tabelas/           # CSVs para o relatório
```

---

## Convenções

- **Português** nas explicações, nos comentários e nos nomes de variáveis.
- Todo número que for para o relatório precisa ter saído de código executado. Nada de valor estimado ou lembrado.
- Toda figura vai para `resultados/figuras/` em PNG, `dpi=140`, com `bbox_inches='tight'`.
- Toda tabela vai para `resultados/tabelas/` em CSV.
- Os módulos em `src/` são a fonte da verdade. Os notebooks importam de lá em vez de redefinir funções.
- `dados/` e `modelos/` estão no `.gitignore` — arquivos grandes ficam no Drive, com o link no README.

---

## Estado atual

Consulte sempre o `DIARIO.md`. É lá que fica o registro do que já foi feito e das métricas obtidas.
