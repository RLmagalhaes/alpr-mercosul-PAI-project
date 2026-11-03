# Armadilhas conhecidas

## Colab

- `colab exec` tem timeout padrão de 30 s. Passe `--timeout 600` em scripts longos.
- `colab upload` falha com 500 de forma intermitente. O contorno é mandar o conteúdo
  por stdin, com `open(caminho, 'w').write(...)`.
- `colab drivemount` exige autenticação interativa, então o Raphael precisa rodá-lo
  no terminal dele. Disparado por um agente, dá `ValueError: mount failed`.
- O kernel mantém os módulos importados na memória. Depois de atualizar `src/` na VM,
  rode `colab restart-kernel`; sem isso, o código antigo continua rodando sem erro.
- A sessão gratuita cai sem aviso. Copie o `last.pt` para o Drive a cada 3 minutos e
  retome com `resume=True`, como fazem os scripts `treinar.py`.

## Modelos

- `chars_best.pt` é um checkpoint de treino da época 39 de 40, com o otimizador
  incluído. Ele tem o mesmo tamanho do `last.pt` no Drive, então o nome "best" pode
  não ser exato.
- Os checkpoints foram treinados com Ultralytics 8.4 e carregam na 8.3.0 fixada em
  `requirements.txt`. Essa combinação foi conferida com o resultado 0,719 / 0,333.

## Dados

- O dataset `trafficbr` tem caixas `plate` que são carros inteiros (45-63% da imagem).
  Filtre por razão largura/altura entre 1,8 e 5,0 e por no máximo 25% da imagem.
- A chave do Roboflow nos commits `2dfa670`, `b2f6457`, `f03f9b5` e `235e06d` foi
  revogada, e usá-la dá `401`. Gere uma nova e forneça por `ROBOFLOW_API_KEY` ou por
  `/content/.roboflow_key` na VM.

## Gabarito

- `scripts/leitor/montar_gabarito.py` sobrescreve o gabarito já transcrito.
- Transcreva `O` e `0` a olho, sem usar a regra de formato. O sistema avaliado usa essa
  regra, e o gabarito ficaria circular, inflando a acurácia.

## Treino

- Rode um ensaio de 3 épocas antes de qualquer treino longo, para que erros de caminho
  apareçam em minutos.
