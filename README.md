# Text Summarizer Seq2Seq (TensorFlow / Keras)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pipeline de sumarização abstrativa em inglês usando dataset Gigaword com modelo Encoder–Decoder (LSTM) + camada de Atenção Keras. Todo o fluxo demonstrado em dois notebooks:

- notebooks/01_preprocessing.ipynb
- notebooks/02_training.ipynb

## 1. Visão Geral do Fluxo
1. Pré-processamento:
   - Carrega Gigaword via tensorflow_datasets.
   - Limpeza simples (regex de espaços).
   - Injeta marcadores <sos> e <eos> nos resumos.
   - Cria duas camadas TextVectorization (documento e resumo).
   - Adapta vocabulários e salva:
     - Dataset vetorizado (tf.data.Dataset.save) em vectorized_gigaword_ds
     - Vocabulários *.npy
     - Pipelines tv_doc_model.keras / tv_sum_model.keras.
2. Treinamento:
   - Carrega pipelines (TextVectorization serializados).
   - Define parâmetros: MAXLEN_DOC=100, MAXLEN_SUM=30, VOCAB_IN=20000, VOCAB_OUT=12000, BATCH_ORIG=16384.
   - Modelo encoder: Embedding + LSTM (return_sequences, return_state).
   - Modelo decoder: Embedding + LSTM + Attention + Concatenate + TimeDistributed Dense (softmax).
   - Loss customizada (SparseCategoricalCrossentropy) com máscara para padding e token eos.
   - Treino com EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
   - Mixed precision habilitada (mixed_float16) e XLA JIT (tf.config.optimizer.set_jit(True)).
3. Inferência:
   - Separação de encoder_model e decoder_model.
   - Implementação de beam_search_decode (suporte a min_len, length penalty alpha, beam_width).
4. Avaliação:
   - Cálculo de ROUGE (rouge_scorer) sobre amostras (val_ds ou arquivo JSON).

## 2. Principais Características Implementadas
- Tokenização interna via TextVectorization (sem SentencePiece).
- Vocabulários distintos para entrada e saída.
- Marcadores <sos>/<eos>.
- LSTM encoder/decoder com Attention (keras.layers.Attention).
- Beam search simples com penalização de comprimento.
- Loss mascarada para ignorar padding e eos.
- Mixed precision + XLA (opcional, já ativado no notebook).
- Salvamento de modelos e vocabulários reutilizáveis.

## 3. Arquitetura
Encoder:
- Input (batch, MAXLEN_DOC)
- Embedding (vocab_size_input, embedding_dim=128)
- LSTM(units=256, return_sequences=True, return_state=True)

Decoder:
- Input sequência alvo deslocada (MAXLEN_SUM+1)
- Embedding (vocab_size_output, 128)
- LSTM (usa state_h/state_c do encoder)
- Attention(dec_outputs, encoder_outputs)
- Concat(dec_outputs, contexto) -> Dense TimeDistributed (softmax)

Loss: SparseCategoricalCrossentropy (from_logits=False) + máscara.
Métrica auxiliar: sparse_categorical_accuracy (token-level).

## 4. Parâmetros Centrais (atual notebook)
MAXLEN_DOC=100  
MAXLEN_SUM=30 (+2 tokens especiais na vetorização)  
VOCAB_IN=20000  
VOCAB_OUT=12000  
embedding_dim=128  
units=256  
BATCH_ORIG=16384 (lote grande antes de rebatchear para treino em 32)  

## 5. Dependências
- tensorflow >= 2.12 (com suporte mixed precision)
- tensorflow_datasets
- rouge-score
- numpy, pandas
- matplotlib
- tqdm (opcional)

Instalação rápida:
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow tensorflow_datasets rouge-score numpy pandas matplotlib
```

## 6. Estrutura de Pastas (Recomendada)
```
text_summarizer_seq2seq/
  data/
    rouge_test_samples.json
  notebooks/
    01_preprocessing.ipynb
    02_training.ipynb
  README.md
```

## 7. Execução do Pré-Processamento (Notebook 01)
Passos principais mostrados:
- Carregar Gigaword: tfds.load('Gigaword', split='train', shuffle_files=True)
- Normalizar espaços e adicionar <sos>/<eos>.
- Adaptar TextVectorization (documentos e resumos).
- Salvar dataset vetorizado: tf.data.Dataset.save(...)
- Salvar vocabulários (.npy) e modelos Keras contendo as camadas de vetorização.

## 8. Treinamento (Notebook 02)
Fluxo:
- Carrega tv_doc_model.keras / tv_sum_model.keras.
- Constrói encoder_model e seq2seq end-to-end.
- Prepara dataset: unbatch -> map para (enc_in, dec_in) / dec_target -> split train/val (VAL_SPLIT=0.2).
- Compila com optimizer Adam(lr=1e-4), masked_loss.
- Callbacks: EarlyStopping(patience=5), ReduceLROnPlateau, ModelCheckpoint.
- Após treino: salvar seq2seq_model.keras, gerar curvas de loss/accuracy.

## 9. Inferência / Beam Search
Implementado no notebook 02 (função beam_search_decode):
- Converte texto bruto via tv_doc.
- Gera tokens iterativamente.
- Controla min_len para evitar término precoce.
Ajustes sugeridos:
- beam_width padrão 5.
- alpha (length penalty) ~0.6; ajustar conforme tamanho médio de resumos.

Uso típico dentro do notebook:
```
pred = beam_search_decode(raw_text, beam_width=5)
print(pred)
```

## 10. Avaliação ROUGE
- Usa rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True).
- Calcula médias F1 sobre lote de validação ou conjunto custom.

## 11. Métricas Disponíveis
- ROUGE-1 / ROUGE-2 / ROUGE-L (F1).
- Acurácia token-level (apenas diagnóstica; não substitui ROUGE).
Futuro: adicionar cobertura, repetição média, comprimento normalizado.

## 12. Exportação e Salvamento
- seq2seq_model.keras: modelo completo (treino).
- best_seq2seq.h5: checkpoint melhor val_loss.
- tv_doc_model.keras / tv_sum_model.keras: pipelines de vetorização (facilitam inferência sem redefinir vocabulário).
Para deployment simplificado basta carregar tv_* + encoder_model + decoder_model.

## 13. Mixed Precision e XLA
Ativado no notebook:
```
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)
```
Observação: Verificar GPU compatível (Ampere+). Caso instabilidade, desativar XLA ou política.

## 14. Limitações Atuais
- Apenas LSTM + Attention padrão (sem Transformer ou coverage).
- Sem regularização avançada (label smoothing, dropout explícito nas camadas).
- Sem scheduler custom de learning rate.
- Beam search sem bloqueio de n-gram repetido.

## 16. Roadmap
- [ ] Adicionar dropout e label smoothing.
- [ ] Implementar no_repeat_ngram_size no beam search.
- [ ] Métrica adicional: perplexity (armazenar logits antes de softmax).
- [ ] Suporte a validação incremental durante treino (callback custom).
- [ ] Exportação para TF SavedModel/TF Serving.
- [ ] Adicionar script CLI (inferência fora do notebook).
- [ ] Fine-tuning com LoRA (quando migrar para Transformer).

## 17. Contribuição
1. Criar branch feature/nome.
2. Atualizar notebooks ou migrar lógica para scripts (src/).
3. Adicionar testes (ex: verificação de beam search vs greedy).
4. Abrir PR descrevendo mudanças.

## 18. Licença
Este projeto está licenciado sob a MIT License (veja arquivo LICENSE).  

Resumo dos direitos:
- Uso, cópia, modificação, distribuição e sublicenciamento permitidos.
- Obrigatório manter o copyright e texto da licença.
- Software fornecido "AS IS", sem garantias.

## 19. Referências
- Keras Seq2Seq + Attention (exemplos oficiais)
- Gigaword (TFDS)
- ROUGE Score Package

## 20. Disclaimer
Os notebooks representam a fonte da verdade do pipeline atual; README será atualizado conforme novas features forem migradas para scripts modulares.

Bom desenvolvimento.
