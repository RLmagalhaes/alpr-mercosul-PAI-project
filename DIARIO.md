# Diário do projeto — ALPR Mercosul

> Atualizado ao fim de **toda** sessão pelo agente. É a memória do projeto entre os dias.
> Ao iniciar uma sessão, leia este arquivo antes de qualquer outra coisa.

**Prazo de entrega:** menos de 1 semana a partir de 03/09 (prazo real da professora é ~1 mês após o fim das aulas, mas o Raphael está atrasado em relação ao roteiro de 7 dias).
**Onde parei:** Dia 4 concluído — CNN de 36 classes treinada, acurácia de teste 0,9435 (com peso de classe balanceado). Pronto pra começar o Dia 5 (pipeline fim a fim + regra do formato).

**Decisão de escopo (registro interno, não sai no relatório pra professora):** dado o atraso, a entrega vai fechar o pipeline completo de visão computacional (detecção → pré-processamento → segmentação → CNN de caracteres → regra de formato), cortando a parte de produção (Dia 6: ONNX, API, Docker, latência) e os slides. Isso **não aparece como corte** em nenhum material voltado à professora (RELATORIO.md, README.md, notebooks) — lá o projeto entregue é descrito como se fosse o escopo original. `api/`, `Dockerfile` e as dependências de ONNX continuam no repo, sem uso, como continuação pessoal de aprendizado do Raphael depois da entrega. Ver plano completo em `/Users/raphaelmagalhaes/.claude/plans/eu-j-estou-atrasado-groovy-squirrel.md`.

---

## Painel de métricas

Preencher conforme os números forem saindo. Estes são os valores que vão para o relatório.

| Métrica | Meta | Obtido | Dia |
| --- | --- | --- | --- |
| mAP@0.5 (detecção, teste) | > 0,90 | 0,992 | 2 |
| mAP@0.5:0.95 (detecção, teste) | — | 0,834 | 2 |
| Precisão / recall (detecção, teste) | — | 0,978 / 0,977 | 2 |
| Acurácia por caractere (CNN, teste) | > 0,95 | 0,9435 | 4 |
| Acurácia por placa — CNN sozinha | — | — | 5 |
| Acurácia por placa — CNN + regra | > 0,80 | — | 5 |
| Latência Keras (p95) | — | — | 6 |
| Latência ONNX (p95) | — | — | 6 |

---

## Dia 0 — Preparação ✅

**Datasets**

- Caracteres: project-swcsj/license-plate-character-extraction v2 (Roboflow)
  - 36 classes (0-9, A-Z), 4711/144/36 imagens, 640x640 px
  - Mediana de 7 caixas por imagem = recortes de placa com caracteres anotados
  - ~33 mil caracteres rotulados no total
  - PENDÊNCIA: redividir 70/15/15 POR IMAGEM (split original é 97/3/0.7)
  - ATENÇÃO: imagens em 640x640, verificar se os caracteres estão deformados

- Detecção: trafficbr/vehicle-plate-color v2 (Roboflow) — 12780/960/257 imagens, classe única "plate", fotos de veículo inteiro (placas Mercosul BR). Salvo em `dados/deteccao/vehicle-plates/`.

**Feito**

- Repositório criado com a estrutura de pastas e os módulos base em `src/`.
- `CLAUDE.md` e `DIARIO.md` criados.
- Roteiro completo em `docs/ROTEIRO_ALPR_7_DIAS.md`.

**Pendências antes do Dia 1**

- [x] Colab CLI testada (`uv tool install google-colab-cli` + `colab new --gpu t4`) — sessão T4 criada e parada com sucesso.
- [x] Dataset de caracteres baixado (ver `dados/caracteres/` acima).
- [x] RodoSol-ALPR: **decisão** — não solicitar, sem tempo pra esperar a liberação (1-5 dias úteis). Projeto segue com o dataset de detecção do Roboflow (`trafficbr/vehicle-plate-color`).
- [x] `git init` e primeiro commit feitos (`58e53e2`).
- [x] `pytest tests/` — 14 testes passaram.

**Próximo passo:** Dia 1 — ambiente, dados e detector treinado.

---

## Dia 1 — Ambiente, dados e detector treinado ✅

**Objetivo:** um modelo YOLO que encontra placas em fotos.

**O que foi feito**
- Ambiente Colab configurado via Colab CLI (`colab new -s dia1 --gpu T4`), Drive montado, ultralytics instalado.
- Dataset de deteccao baixado direto no Colab (trafficbr/vehicle-plate-color v2), salvo local na VM (`/content/dados/deteccao`) em vez do Drive — escrever milhares de arquivos pequenos direto no Drive trava o kernel (mount em rede).
- Inventario do dataset gerado (ver tabela abaixo) e galeria de 6 amostras com bbox conferida visualmente — anotacoes corretas.
- Ensaio de 3 epocas rodado com sucesso (script `notebooks/treinar_ensaio.py`, via subprocess em background pra nao travar o CLI em treinos longos).
- Treino completo (40 epocas, `yolo11n.pt`, `patience=10`) rodado via `notebooks/treinar_detector.py`, com checkpoint periodico (`last.pt` copiado pro Drive a cada 180s) e retomada automatica (`resume=True`) apos quedas de sessao — a sessao gratuita do Colab caiu varias vezes ao longo do treino, mas nenhum progresso foi perdido gracas ao checkpoint. Treino concluido nas 40 epocas configuradas, pesos finais copiados pra `modelos/detector` no Drive.

**Métricas obtidas**
- Inventario: train 12780 imagens/13386 caixas (area media da placa 5,73% da foto), valid 960/995 (4,97%), test 257/268 (5,86%).
- Ensaio (3 epocas, yolo11n.pt) — preliminar, so pra validar que o pipeline funciona:
  - epoca 3: mAP50 = 0,99 · mAP50-95 = 0,773 · precisao = 0,980 · recall = 0,954
- **Treino completo (40 epocas, yolo11n.pt) — resultado final:**
  - mAP50 = 0,994 · mAP50-95 = 0,851 · precisao = 0,987 · recall = 0,979
  - val/box_loss = 0,611 · val/cls_loss = 0,299 · val/dfl_loss = 0,860 (losses de val abaixo das de treino, sem sinal de overfitting nesse ponto)
  - LIMITACAO HONESTA: o disco local do Colab (`/content`) e apagado a cada sessao nova, entao o `results.csv`/`results.png` finais so guardam a ULTIMA sessao de treino (1 epoca, a de numero 40), nao a curva completa das 40 epocas. Os pesos sao cumulativos e reais (carregam o aprendizado de todas as sessoes via `resume=True`), so o historico visual da curva epoca-a-epoca ficou fragmentado entre sessoes. Nao afeta a validade do modelo final, so a rastreabilidade da evolucao.

**Decisões**
- Dataset de deteccao: `trafficbr/vehicle-plate-color` v2 (Roboflow) em vez do sugerido originalmente no roteiro — 1 classe ("plate"), fotos de veiculo inteiro, placas Mercosul BR.
- Modelo base: `yolo11n.pt` (nano), como no roteiro.
- Dados brutos (imagens) ficam no disco local da VM, nao no Drive — persistem so durante a sessao, mas o download do Roboflow e rapido o suficiente pra refazer a cada sessao nova. So os resultados (pesos, figuras, tabelas) vao pro Drive.
- RodoSol-ALPR descartado (ver Dia 0) — segue so com este dataset.
- Estrategia de resiliencia do treino: `save_period=1` + thread em background copiando `last.pt` pro Drive a cada 180s + deteccao automatica de checkpoint no inicio do script (retoma com `resume=True` se achar checkpoint no Drive) — necessario porque a sessao gratuita do Colab caiu repetidas vezes durante o treino de 40 epocas.

**Pendências**
- Nenhuma pendencia do Dia 1 — treino completo, pesos e curvas conferidos (ver limitacao do historico do `results.csv` acima).

**Próximo passo**
- Dia 2 — avaliar o detector no conjunto de TEST (nao so o val visto no treino) e recortar as placas detectadas, alimentando o dataset de caracteres do Dia 3.

---

## Dia 2 — Avaliação da detecção e recorte das placas ✅

**Objetivo:** métricas confiáveis do detector e as placas recortadas para o Dia 3.

**O que foi feito**
- Detector avaliado no conjunto de TESTE (257 imagens, 268 instâncias) — não só no val visto durante o treino.
- IoU implementado do zero (`iou()`) e testado com casos de sanidade (caixas idênticas, sem sobreposição, sobreposição parcial conhecida).
- Efeito do limiar de NMS observado numa imagem com múltiplas placas: limiares até 0,7 deram 2 detecções corretas; limiar 0,9 (muito permissivo) deixou passar uma caixa duplicada da mesma placa (3 detecções).
- Análise de erros: comparação predição x anotação via IoU, ranqueando os piores casos.
- Todas as placas recortadas a partir das caixas **anotadas** (não as previstas), com margem de 8%, reaproveitando a mesma lógica de `src/preprocessamento.recortar()`. Rodado direto no disco local da VM — **não fica no Drive** (ver decisão de gestão de espaço abaixo).
- Limpeza de espaço no Drive: apagados os 23 checkpoints por época do detector (`epoch*.pt`, `last.pt` — mantido só `best.pt`), a pasta `detector_checkpoint/` (mecanismo de resume do Dia 1, sem uso depois que o treino terminou) e `dados/deteccao/` (dataset bruto, que por engano tinha ficado no Drive além da VM). Pasta do projeto no Drive caiu de 1,2 GB para 70 MB. **Pendência do próprio Raphael:** esvaziar a lixeira do Drive pela interface, senão o espaço não é liberado de verdade.
- Achado à parte: existia um `placas_recortadas.zip` no Drive (datado de 29/08) com números muito parecidos aos nossos, mas sem nenhum registro de que código o gerou. Decisão: não reaproveitar — regerado do zero de forma rastreável (código no notebook, resultado com contagem de problemas). O resultado bateu (e superou um pouco, por causa da conversão de polígono — ver abaixo).

**Métricas obtidas**
- mAP@0.5 = 0,992 · mAP@0.5:0.95 = 0,834 · precisão = 0,978 · recall = 0,977 (conjunto de teste).
- 2 de 257 placas não detectadas (IoU=0); os piores casos entre as detectadas têm IoU muito baixo (0,01-0,03), ou seja, são erros grosseiros, não só imprecisão de borda.
- Recorte de todas as placas anotadas: 13.725 recortadas no total (train 12.539, valid 941, test 245). 720 anotações vieram em formato de polígono e foram convertidas pra caixa delimitadora (mesma conversão que o Ultralytics já faz sozinho). 924 recortes descartados por ficarem abaixo do tamanho mínimo (30x10px).
- **Problema de dados registrado (não contornado):** 76% dos recortes aceitos (10.420 de 13.725) são "muito pequenos" (altura<40px ou largura<100px) — ver `resultados/figuras/recortes_placas_tamanhos.png`. Nos exemplos pequenos, o texto da placa já sai borrado só pela miniatura, antes de qualquer pré-processamento. É uma limitação do dataset de origem (fotos de resolução/enquadramento variados), não um bug do recorte. Efeito esperado: a segmentação de caracteres do Dia 3 deve ter uma taxa de falha bem maior nesse subconjunto — isso precisa aparecer explícito na análise de erros do Dia 3/4, sem tentar mascarar com upscaling ou heurística de salvamento.

**Decisões**
- Placas recortadas ficam só no disco local da VM (`/content/dados/placas_recortadas/`), nunca no Drive — mesma lógica já usada pro dataset bruto no Dia 1. Como o recorte é rápido (poucos minutos, sem GPU), a célula 2.5 deve ser rodada de novo no início da sessão do Dia 3 em vez de tentar persistir os arquivos entre sessões.
- Anotações em polígono são convertidas pra bounding box (min/max), não descartadas — mantém mais dados e é consistente com o que o Ultralytics já faz na validação.

**Pendências**
- Esvaziar a lixeira do Google Drive (ação manual do Raphael, fora do alcance da CLI).

**Próximo passo**
- Dia 3 — rodar de novo a célula de recorte (2.5) no início da sessão, depois pré-processamento e geração do dataset de caracteres em `chars/<CLASSE>/`.

---

## Dia 3 — Pré-processamento e dataset de caracteres ✅

**Objetivo:** funções de tratamento prontas e a pasta `chars/` gerada.

**O que foi feito**
- Funções de tratamento (`endireitar`, `preparar`, `detectar_layout`, `projecao_vertical`, `segmentar`) usadas diretamente de `src/preprocessamento.py` — enviadas pra VM via `colab upload` em vez de redefinidas no notebook (corrige uma inconsistência que vinha do Dia 2, onde o `iou()` tinha sido redefinido inline).
- Demonstração do pré-processamento (recorte → cinza → CLAHE → Otsu) numa placa bem alinhada do teste.
- Demonstração da segmentação em 7 fatias, comparando um caso bem alinhado com um caso de perspectiva forte (ver limitação abaixo).
- Dataset de caracteres baixado (`project-swcsj/license-plate-character-extraction` v2 — **dataset separado** do de detecção, já vem com cada caractere anotado individualmente) e convertido em `chars/<CLASSE>/` com CLAHE+Otsu aplicado a cada recorte.
- Checagem visual de amostras aleatórias de 8 classes para conferir se os recortes batem com o rótulo.

**Métricas obtidas**
- Recorte de placas (Dia 2, regerado): 13.735 no total (train 12.548, valid 942, test 245 — pequena variação frente ao rodado no Dia 2 por causa da ordem de leitura do glob, sem impacto real).
- Dataset de caracteres: **31.718 caracteres rotulados** no total (train 30.530, valid 958, test 230), em 36 classes.
- Classes ignoradas por não pertencerem ao alfabeto de placa: `EUR` (1.708 casos) e `-` (44 casos) — são marcações do dataset original (bandeira/hífen de formato antigo), corretamente filtradas.
- Classes mais raras no treino: **Q (120)** e **O (228)** — bem abaixo do resto (a maioria das letras fica entre 400-1500, dígitos entre 1000-2100). Como Q e O também são visualmente parecidos com 0, esse é o par mais provável de confusão pro Dia 4.

**Problemas de dados registrados (não contornados)**
1. **Perspectiva forte quebra a segmentação:** `endireitar()` só corrige rotação no plano (via `minAreaRect`), não perspectiva. Numa foto tirada de ângulo mais agressivo, a segmentação por projeção (que assume fatias de largura igual) sai completamente desalinhada — ver `resultados/figuras/segmentacao.png`, que mostra lado a lado um caso bom (fatias legíveis) e um caso ruim (ilegível). Corrigir de verdade exigiria os 4 cantos da placa anotados, que este dataset não tem.
2. **Caracteres deformados na origem (pendência do Dia 0, resolvida):** o dataset de caracteres aplicou "Resize to 640x640 (Stretch)" no pré-processamento do Roboflow — ou seja, esticou cada imagem sem preservar a proporção original, distorcendo a forma dos caracteres de um jeito não uniforme (a proporção original não é recuperável). Aceito como limitação do dataset escolhido.
3. **Ruído de rótulo:** checagem visual encontrou pelo menos um exemplo rotulado como "O" que na verdade mostra um "H". Não foi feita filtragem manual (não há tempo nem um método confiável pra achar todos os casos parecidos em ~31 mil imagens) — fica registrado como ruído esperado de dataset anotado em massa, e é uma fonte honesta de erro que a CNN do Dia 4 provavelmente vai herdar.

**Decisões**
- Dataset de caracteres é gerado direto das anotações por caractere (mais preciso), **não** a partir do `segmentar()` sobre os recortes do Dia 2 — `segmentar()` é usado só na hora da inferência (Dia 5), quando não há caixa por caractere disponível.
- `chars/` fica só na VM (não no Drive), mesma lógica do Dia 2 — só o resumo (JSON) e as figuras vão pro Drive.
- Notebooks passam a importar as funções de `src/` de verdade (via upload pontual do pacote pra VM), em vez de redefini-las — alinhado com a convenção do projeto.

**Pendências**
- Nenhuma pendência nova do Dia 3.

**Próximo passo**
- Dia 4 — treinar a CNN de 36 classes sobre `chars/`, com atenção especial à matriz de confusão em torno de Q/O/0.

---

## Dia 4 — CNN de caracteres ✅

**Objetivo:** classificador de 36 classes com acurácia acima de 95%.

**O que foi feito**
- `chars/` regenerado do zero no início da sessão (disco local do Colab é apagado a cada sessão — mesma lógica dos Dias 2/3), com o mesmo código do Dia 3: 31.718 caracteres, 36 classes (train 30.530, valid 958, test 230). Números batem exatamente com o Dia 3.
- CNN pequena treinada do zero (`src`-free, arquitetura só do roteiro: 3 blocos Conv2D+MaxPool, Dense 128, Dropout 0.3), 359.588 parâmetros, `EarlyStopping`/`ModelCheckpoint`/`ReduceLROnPlateau`.
- **Primeira rodada** (sem peso de classe): acurácia de teste = 0,9348, parou em 13 épocas. Matriz de confusão mostrou **Q e O com 0% de acerto** — exatamente as duas classes mais raras no treino já apontadas como risco no Dia 3 (Q=120 exemplos, O=228).
- **Segunda rodada** (com `class_weight` balanceado por classe, pelo inverso da frequência no treino — pesos de 0,4 a 7,1): acurácia subiu para **0,9435**, 26 épocas. O passou de 0% para 100% de recall (mas ganhou falsos positivos vindos do dígito "0"); Q passou de 0% para 33% de recall.
- Matriz de confusão e pares confundidos gerados a partir da 2ª rodada (a que ficou nos artefatos finais).
- **Achado de processo corrigido nesta sessão:** as figuras/tabelas dos Dias 1 a 3 nunca tinham sido trazidas do Drive pro repositório local — só existiam na cópia da VM. Baixadas agora todas de uma vez pra `resultados/figuras/` e `resultados/tabelas/` locais (8 figuras, 9 tabelas/JSONs), senão o relatório final não teria de onde puxar essas imagens. Cuidado nos próximos dias: sempre baixar do Drive pro local antes de fechar a sessão.

**Métricas obtidas**
- Acurácia no teste (final, com peso de classe): **0,9435** — abaixo da meta de 0,95, mas por pouco.
- Perda no teste: 0,485. Parâmetros: 359.588. Épocas até o `EarlyStopping`: 26 (de 30 possíveis).
- Acurácia por classe (36 classes): 32 das 36 classes com recall ≥ 0,75 (a maioria em 1,0). As exceções: **O** (recall 1,0 mas precisão 0,33 — passou a "roubar" o dígito 0), **Q** (recall 0,33, 3 exemplos no teste), **0** (recall 0,56 — pagou o preço pelo peso extra em O), **V** (recall 0,75).
- Ver `resultados/tabelas/classification_report_cnn.csv` e `resultados/tabelas/resumo_dia4.json` para os números completos.
- **LIMITAÇÃO HONESTA:** o conjunto de teste tem só 230 caracteres pra 36 classes — Q e O aparecem 3 e 1 vez, respectivamente. Cada erro nessas classes pesa ~0,4-1,3 ponto percentual na acurácia total. Não dá pra saber se 33%/100% de recall nessas classes é representativo ou só sorte de amostra pequena — decisão tomada foi aceitar o resultado e não continuar ajustando hiperparâmetro em cima do ruído de 1-3 exemplos.

**Pares mais confundidos** (matriz de confusão final, real → previsto)
- 0→O: 2 · 0→4: 1 · 0→P: 1 · 4→L: 1 · 7→1: 1 · 9→3: 1 · 9→S: 1 · B→8: 1 · Q→C: 1 · Q→D: 1 · S→3: 1 · U→J: 1
- A aposta do roteiro (O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6) acertou em parte: 7↔1 e B→8 apareceram; O↔0 apareceu só depois do ajuste de peso (antes disso Q↔0 é que aparecia). S↔5 e Z↔2/G↔6 não apareceram — no lugar, surgiram confusões novas (9→S, U→J, 4→L) que fazem sentido visual mas não estavam na lista original.

**Decisões**
- Usar `class_weight` balanceado (peso = total_treino / (36 × contagem_da_classe)) em vez de oversampling ou data augmentation extra — mudança de uma linha no `fit()`, sem precisar tocar no pipeline de geração de dados.
- Aceitar 0,9435 (abaixo da meta de 0,95) como resultado final do Dia 4 em vez de continuar ajustando: o resíduo está concentrado em classes com 1-3 exemplos no teste, então mais tuning ficaria ajustando em cima de ruído estatístico, não corrigindo um problema real do modelo. Registrado como limitação honesta, não maquiada.
- `modelos/cnn_chars.keras` fica só no Drive (mesma lógica de `modelos/detector` no Dia 1 — `.gitignore` cobre `*.keras`).

**Pendências**
- Nenhuma pendência nova do Dia 4, além da limitação de dados já registrada acima.

**Próximo passo**
- Dia 5 — montar `ler_placa()` fim a fim (detecção → pré-processamento → segmentação → CNN → regra do formato) e medir a acurácia por placa. A regra de formato (posição só aceita letra OU só dígito) deve corrigir de graça os erros tipo 0→O e Q→0/C/D que a CNN sozinha não resolve.

---

## Dia 5 — Pipeline fim a fim e regra do formato

**Objetivo:** `ler_placa()` funcionando e as métricas finais medidas.

**O que foi feito**
_(preencher)_

**Métricas obtidas**
_(preencher — acurácia por caractere e por placa, antes e depois da regra; posição que mais erra)_

**Decisões**
_(preencher)_

**Próximo passo**
_(preencher)_

---

## Dia 6 — ONNX, API e latência

**Objetivo:** o sistema virando serviço.

**O que foi feito**
_(preencher)_

**Métricas obtidas**
_(preencher — latência média/p50/p95 em Keras e ONNX, limiar de confiança escolhido)_

**Decisões**
_(preencher)_

**Próximo passo**
_(preencher)_

---

## Dia 7 — Relatório e slides

**O que foi feito**
_(preencher)_

**Entregues**
- [ ] `RELATORIO.pdf`
- [ ] 10 slides
- [ ] Repositório limpo e público
- [ ] Seção de privacidade escrita

---

## Registro de problemas

Anote aqui o que quebrou e como foi resolvido. Vira a seção "Limitações" do relatório.

| Dia | Problema | Solução |
| --- | --- | --- |
| 1 | `colab exec` tem timeout padrão de 30s, causando `TimeoutError` em scripts mais demorados (matplotlib, downloads, treino) | Sempre passar `--timeout` explícito (600 pra scripts longos, 60 pra snippets rápidos) |
| 1 | `colab upload` falhava de forma inconsistente (`File or directory not found` no caminho remoto) mesmo com sintaxe correta | Escrever o conteúdo do script via stdin em `colab exec` (`open(caminho,'w').write(codigo)`) em vez de usar `colab upload` |
| 1 | Sessões gratuitas do Colab caem imprevisivelmente (perda de conexão, kernel perdido) durante o treino de 40 épocas — uma vez perdendo ~1h de progresso | Checkpoint periódico (`last.pt` copiado pro Drive a cada 180s) + detecção automática e retomada (`resume=True`) no início do script |
| 1 | Disco local do Colab é apagado a cada sessão nova, então `results.csv` só guarda a última sessão de treino, não o histórico completo das 40 épocas | Aceito como limitação conhecida — pesos finais são cumulativos e válidos, só o gráfico de evolução ficou fragmentado |
| 4 | Figuras/tabelas dos Dias 1-3 nunca tinham sido trazidas do Drive pro repositório Git local — só existiam na cópia da VM (`colab exec` salva direto no Drive montado, não no Mac) | Baixadas todas de uma vez via `colab download` no Dia 4; a partir de agora, baixar do Drive pro local antes de fechar cada sessão, não só ao final do projeto |
| 4 | CNN de 36 classes ficou em 0,9348 na primeira rodada (meta > 0,95), com Q e O (as classes mais raras do treino) em 0% de acerto no teste | `class_weight` balanceado por classe no `fit()` — subiu para 0,9435; residual aceito como limitação (Q/O têm só 1-3 exemplos no teste, ruído estatístico) |
