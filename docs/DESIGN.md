# Design — MAGMA Original + MAGMA Narrativa

> Design doc vivo do projeto: escopo, requisitos, arquitetura, decisões com alternativas consideradas, protocolo de avaliação e plano. Qualquer agente deve conseguir continuar o trabalho a partir daqui.
> As specs concretas de cada módulo (interfaces, contratos e trechos de código) derivam deste documento e ficam ao lado dele, em `docs/`.
>
> **Status:** análise concluída. **Decisão principal:** reimplementar o MAGMA do zero, fiel ao artigo (§5).
> **Decididas:** D4 (baselines), D6 (sem framework de LLM), D7 (FAISS exato), D8 (Qwen3-Embedding-0.6B).
> **Pendentes:** D1 (benchmark; em aberto a tarefa do RF1 e o tamanho dos subconjuntos), D2 (roteamento; confirmar se a alternativa citada é JEPA ou GEPA), D3 (LLM local + juiz), D5 (idioma; tende a inglês pela D1), D9 (ontologia como perfis num registro declarativo; recomendação escrita, falta confirmar).
> **Próximo passo:** confirmar D9 → discutir D1 → fechar D2, D3, D5 → M0 (§10). Decisões discutidas uma por vez.
> **Regra da §9:** toda decisão registra recomendação, justificativa, alternativas comparadas e por que cada uma foi descartada (material direto para o artigo). Sem tom de roteiro de apresentação.
> **Regra de escopo (PoC):** implementar tudo o que o artigo descreve e tudo o que a extensão exige; só otimizações e extras vão para a §12.
>
> Mapa: §0 escopo · §1 artigo · §2 requisitos · §3 ideias avaliadas · §4 auditoria do código oficial · §5 decisão ·
> §6 Branch A · §7 Branch B · §8 benchmark/métricas · §9 pendências · §10 milestones · §11 referências ·
> §12 fora do escopo do PoC

---

## 0. Escopo e princípios

**Entregas**
1. **Branch `magma-original`** — reimplementação fiel do MAGMA (Jiang et al., ACL 2026).
2. **Branch `magma-narrative`** — extensão do MAGMA para narrativas multiagente (fato × declaração, visibilidade, contradição).
3. **Benchmark comparativo** entre as duas arquiteturas e baselines tradicionais de memória.

**Princípios**
- **Fase atual: prova de conceito (PoC).** Roteiro: PoC → MVP → implementação final. O PoC implementa **todos os componentes descritos no artigo** (router de intenção, RRF, beam search, linearização com budgeting, dual-stream, 4 grafos, persistência) e **todos os da extensão** (RF1–RF3 + agentes e orquestrador). Ficam para depois só otimizações e extras que nem o artigo nem os requisitos pedem (§12).
- **Baixa complexidade:** implementar só o necessário para os requisitos (§2); o resto vira trabalhos futuros.
- **Reprodutibilidade:** `temperature=0`, `seed` fixo, cache de chamadas LLM em disco, hiperparâmetros num único arquivo.
- **Modelos locais por padrão** via endpoint OpenAI-compatível (ex.: Ollama); provedores remotos são opcionais (ex.: só para o juiz).
- **Evidência principal vem de benchmarks públicos e estabelecidos** (validados e citados na literatura), não de dados criados pelo próprio projeto. Conjunto próprio só como diagnóstico complementar (D1).
- **LoCoMo/LongMemEval** não têm narrativa multiagente, segredos nem mentiras; servem para fidelidade da Branch A e para a retenção (H2), não para os RFs narrativos.
- **Roteamento de intenção:** em aberto (D2).

---

## 1. Artigo MAGMA — o que é "a arquitetura original"

Referência: Jiang, D.; Li, Y.; Li, G.; Li, B. *MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents*. ACL 2026 (Long Papers), pp. 36848–36865. https://aclanthology.org/2026.acl-long.1709/ · Código oficial: https://github.com/FredJiang0324/MAGMA (MIT).

### 1.1 Visão geral (3 camadas)
1. **Query Process** — Intent-Aware Router → Adaptive Topological Retrieval → Context Synthesizer.
2. **Data Structure Layer** — Vector DB + 4 grafos de relação (Semântico, Temporal, Causal, Entidade).
3. **Write/Update Process** — dual-stream: *Fast Path* (Synaptic Ingestion, síncrono, sem LLM) + *Slow Path* (Asynchronous Consolidation, LLM).

### 1.2 Estrutura de dados
- Multigrafo direcionado variante no tempo `G_t = (N_t, E_t)`.
- **Event-Node** `n_i = <c_i, τ_i, v_i, A_i>`: conteúdo, timestamp discreto, embedding denso, atributos (entidades, pistas temporais, descritores).
- Arestas particionadas em 4 subespaços:
  - **Temporal** `(n_i, n_j)` com `τ_i < τ_j`: cadeia imutável (backbone).
  - **Causal**: direcionada, existe se `S(n_j | n_i, q) > δ`; inferida pela consolidação (LLM). Suporta "Why".
  - **Semântica**: não direcionada, `cos(v_i, v_j) > θ_sim`.
  - **Entidade**: liga eventos a **nós de entidade abstratos** (object permanence).
- O artigo menciona hierarquia (eventos atômicos → agrupamentos episódicos), mas não a detalha.

### 1.3 Query (Algoritmo 1 — Heuristic Beam Search)
- **Stage 1 – Análise da query:** classificador leve de intenção `T_q ∈ {WHY, WHEN, ENTITY}`; parser temporal → janela `[τ_s, τ_e]`; embedding denso `v_q` (vetor da query) + keywords esparsas `q_key`.
- **Stage 2 – Âncoras via RRF:** `S_anchor = TopK( Σ_{m∈{vec,key,time}} 1/(k + r_m(n)) )`, `k=60`.
- **Stage 3 – Travessia adaptativa:**
  `S(n_j|n_i,q) = exp( λ1·φ(type(e_ij),T_q) + λ2·sim(v_j, v_q) )`, com `φ(r,T_q) = w_Tqᵀ · 1_r`.
  `score_v = score_u·γ + s_uv` (decaimento γ). Mantém top-`BeamWidth` por profundidade; para em `MaxDepth` ou `Budget`.
- **Stage 4 – Linearização:**
  1. Ordenação: WHEN → por timestamp; WHY → sort topológico nas arestas causais.
  2. Scaffold com proveniência: `[<t:τ_i> conteúdo <ref:id>]` concatenados (Eq. 7).
  3. Token budgeting por saliência: nós de baixo score viram "...3 intermediate events...".

### 1.4 Escrita
- **Fast path (Alg. 2):** segmenta evento → aresta TEMP com o último nó → embedding → indexa no VDB → enfileira id. **Sem LLM.**
- **Slow path (Alg. 3):** worker desenfileira → vizinhança 2-hop → prompt → LLM infere arestas **causais e de entidade** → adiciona.

### 1.5 Hiperparâmetros (Tabela 6)
| Módulo | Parâmetro | Valor |
|---|---|---|
| Embedding | modelo | all-MiniLM-L6-v2 (384d) / text-embedding-3-small (1536d) |
| LLM | backbone | gpt-4o-mini, temp 0.0 |
| Retrieval | RRF k | 60 |
| | Vector Top-K | 20 |
| | w_keyword (fusão) | 2.0–5.0 |
| | Sim. threshold | 0.10–0.30 |
| Traversal | Max depth | 5 hops |
| | Max nodes | 200 |
| | Drop threshold | 0.15 |
| Pesos | λ1 | 1.0 |
| | λ2 | 0.3–0.7 |
| | w_entity | 2.5–6.0 |
| | w_temporal | 0.5–4.0 |
| | w_causal | 3.0–5.0 |
| | w_phrase | 2.5–5.0 |

### 1.6 Prompts (Apêndice C)
- **C.1 Event Extraction (JSON):** `entities`, `topic`, `relationships`, `semantic_facts`, `dates_mentioned`, `summary`. Entrada: speaker, text, prev_summary.
- **C.2 Adaptive QA:** responde só com o contexto; "Information not found" se ausente; instrução dinâmica por categoria (Multi-hop / Temporal / Open-domain / Single-hop).
- **C.3 Judge:** score contínuo [0,1] (1.0/0.8/0.6/0.4/0.2/0.0), saída JSON `{score, reasoning}`.

### 1.7 Resultados de referência
- LoCoMo (judge, gpt-4o-mini): MAGMA **0.700** vs Nemori 0.590, A-MEM 0.580, MemoryOS 0.553, Full Context 0.481.
- LongMemEval: 61.2% médio (0.7k–4.2k tokens/query vs 101k do full context).
- Eficiência (Tab. 3): build 0.39h, **3.37k tokens/query**, **1.47s latência**.
- Ablação (Tab. 4): sem Adaptive Policy 0.637; sem Causal 0.644; sem Temporal 0.647; sem Entity 0.666.

### 1.8 Inconsistências e lacunas do próprio artigo
- **Intenções:** Stage 1 diz `{WHY, WHEN, ENTITY}`, mas o prompt QA usa categorias LoCoMo (Multi-hop/Temporal/Open-domain/Single-hop) "geradas pelo router". Não fica claro se são um ou dois classificadores.
- **`w_phrase`** aparece nos hiperparâmetros, mas não existe aresta "phrase" no texto.
- **δ (limiar causal)**, **γ (decay)** e **BeamWidth** não têm valor declarado.
- **"Event segmentation"** no fast path não é especificada (cada turno? cada fato?).
- **A extração C.1 usa LLM**, mas o fast path diz "sem LLM bloqueante" → a extração precisa ir para o slow path.
- Em benchmark offline, a consolidação "assíncrona" roda na prática em lote após a ingestão.
- O "classificador leve" de intenção não é descrito (regras? modelo? LLM?).
- **Granularidade do evento × janela do embedding:** o embedding padrão (MiniLM) corta qualquer texto acima de 256 word-pieces (~190 palavras) sem avisar. Somado à segmentação de eventos não especificada, isso faz mensagens longas perderem conteúdo silenciosamente na busca semântica. O artigo não discute o tema; os turnos do LoCoMo são curtos, o que esconde o problema. Nesta implementação: a janela de 32k do embedding escolhido (D8) evita o truncamento; a segmentação fica para depois do PoC (§12).

> **Implicação:** "fiel ao artigo" = implementar exatamente as Eqs. 3–7, Algoritmos 1–3, Tabela 6 e prompts C.1–C.3, e **documentar as escolhas** onde o artigo é omisso (§6.4).

---

## 2. Requisitos do projeto (MAGMA Narrativa)

**Objetivo geral:** adaptar o MAGMA para preservar a coerência de narrativas multiagente geradas por LLMs, com enredos longos que contenham **mentiras, segredos e pontos de vista diferentes**. Cada agente conhece a própria identidade (experiências, relações, personalidade).

**Problema:** com agentes agindo de forma independente, a base de conhecimento compartilhada muda o tempo todo. É preciso impedir que dados sigilosos vazem e responder corretamente com tempo e custo aceitáveis.

**Requisitos funcionais**
- **RF1 — Fato × Declaração:** o grafo separa acontecimentos reais de meras declarações dos personagens.
- **RF2 — Visibilidade:** regras de visibilidade nos nós garantem que cada agente acesse **apenas o que presenciou**.
- **RF3 — Contradição:** o sistema detecta contradições no histórico e **atualiza o status** de falas passadas diante de novos fatos.
- **RF4 — Avaliação:** medir coerência narrativa e eficiência computacional contra abordagens tradicionais de memória.

**Hipóteses (resultados esperados)**
- H1: separação fato/declaração + visibilidade evita vazamento de segredos e sustenta pontos de vista individuais.
- H2: a extensão preserva a retenção de longo prazo das camadas originais do MAGMA.
- H3: menor consumo de tokens e menor latência que memórias tradicionais (full context).

**Requisitos metodológicos**
- Comparar contra o MAGMA original **e** contra uma implementação RAG de terceiros (não só contra si mesmo).
- Variáveis independentes (arquitetura de memória, presença de visibilidade/status/contradição) × dependentes (vazamento, consistência, tokens, latência) explícitas.
- Grupo controle × grupo experimental; tratamento estatístico dos resultados.
- Justificar cada ferramenta escolhida e as alternativas descartadas.

**Base teórica:** MAGMA (Jiang et al., 2026); AdaMARP (Xu et al., 2026) — ação física × fala; DEBATE (Chuang et al., 2025) — consistência de opinião em role-play; ToMAgent (Zhou et al., 2024) — quem sabe o quê; *Memory is Reconstructed, Not Retrieved* (Ji et al., 2026) — grafo vs texto linear em custo.

---

## 3. Ideias avaliadas e decisões de escopo

### 3.1 Ideias de design consideradas
- **Problemas motivadores:** *Graph Poisoning* (mentira gravada como fato), *Epistemic Leakage* (agente usa informação que não presenciou), *colapso causal* (alegações refutadas não são invalidadas).
- **Comparativo de abordagens:** A) Lorebook + sliding window (baseline tradicional); B) RAG vetorial + MAGMA vanilla; C) MAGMA Epistêmico (proposta).
- **Ontologia de 5 nós primitivos:** `Agent`, `Entity`, `Event`, `PhysicalAction`, `Claim`, com ~20 tipos de aresta em 4 famílias (epistêmica, causal, temporal, física).
- **Dual-Facet Event:** cada turno gera 1 `Event` ligado a uma faceta `PhysicalAction` (o que se viu) e uma faceta `Claim` (o que foi dito, `epistemic_status: UNVERIFIED`).
- **Ciclo de vida da mentira:** `STATED` → `BELIEVES` → `MOTIVATES` → evidência `INVALIDATES` → `epistemic_status = REFUTED_LIE`.
- **Controle de visibilidade:** `visible_to: [agentes]` nos nós; filtro antes da travessia.
- **Cenário de teste:** *Roubo do Protótipo na Estação Orbital Épsilon-4*, 4 agentes (vítima, ladrão mentiroso, testemunha com segredo próprio, auditor/investigador). Cenário sintético fechado → sem contaminação de dados.
- **Métricas:** ELR (Epistemic Leakage Rate), DCR (Deception Consistency Rate), CIA (Causal Invalidation Accuracy), latência + tokens.

### 3.2 Triagem

| Ideia | Decisão | Motivo |
|---|---|---|
| Separação `Event`/`PhysicalAction` × `Claim` | **MANTER (núcleo)** | É o RF1 |
| `visible_to` nos nós + filtro na recuperação | **MANTER (núcleo)** | É o RF2 |
| `INVALIDATES` / `CONTRADICTS` + `epistemic_status` | **MANTER (núcleo)** | É o RF3 |
| `STATED` (autoria) | **MANTER** | Necessário para saber quem disse o quê (e quem mentiu) |
| `BELIEVES {confidence}` | **SIMPLIFICAR** | Status relativo ao observador (§7.4) em vez de grafo de crença separado |
| Cenário Épsilon-4 com 4 agentes | **MANTER** | Base do benchmark narrativo próprio |
| ELR, DCR, CIA, tokens, latência | **MANTER** | Ligadas diretamente aos RFs |
| 5 tipos de nó primitivos | **CATÁLOGO** (D9) | Perfil medido no PoC (`narrative-min`): `Event` (fato observado; ação física em `action_text`), `Claim` (declaração), `Entity` (inclui agentes, `kind=agent`). Os outros primitivos ficam no catálogo de candidatos da D9 e entram num perfil enriquecido se a análise de erros pedir |
| ~20 tipos de aresta | **CATÁLOGO** (D9) | 9 tipos no `narrative-min` (§7.2); as demais ficam no catálogo da D9 |
| `REPLIES_TO`, `ADDRESSED_TO` | **CORTAR** | Não exigidos pelos RFs |
| `SUPERSEDES` | **MANTER (núcleo)** | Distingue atualização legítima ou mudança de versão do autor de contradição entre autores (D9); exigida pelos benchmarks de atualização do RF3 |
| Sarcasmo, boatos, promessas, ToM de 2ª ordem, decaimento, hipotéticos, vínculos sociais, autoridade | **CORTAR** → trabalhos futuros | Cada um é um projeto à parte |
| Lorebook / Seed Graph | **MÍNIMO** | Só um JSON de cenário (personagens, segredos, fatos iniciais com `visible_to`) |
| Neo4j / Memgraph / Cypher | **CORTAR** | Infra extra; o artigo (§3.5) admite backend em memória. NetworkX + índice vetorial em memória: zero infra, reprodutível, mesmo backend nas duas branches (variável controlada) |
| FastAPI / .NET | **CORTAR** | Não precisa de servidor: CLI + scripts de benchmark |
| LangChain / LangGraph / PydanticAI no núcleo | **CORTAR** | Abstraem justamente o que é a contribuição (roteamento, recuperação, consolidação, prompts) e atrapalham a medição de tokens/latência (D6) |
| LlamaIndex (ou LangChain) | **SÓ NO BASELINE C2** | É o "RAG de terceiros" exigido; isolado em `baselines/naive_rag.py` |
| LLMs proprietários antigos (GPT-4o, Claude 3.5, Gemini 1.5) | **SUBSTITUIR** | LLM local via endpoint OpenAI-compatível (D3) |

---

## 4. Auditoria do código oficial

Alvo: `FredJiang0324/MAGMA` @ commit `467cb70` (~14k linhas Python). Caminhos abaixo são relativos à raiz desse repositório.

### 4.1 Divergências entre artigo e código (verificadas)

| # | Divergência | Evidência |
|---|---|---|
| 1 | Beam Search do Alg. 1 é código morto | `memory/query_engine.py:1002` define `_probabilistic_beam_search`; **zero chamadas** |
| 2 | Classificador `{WHY,WHEN,ENTITY}` é código morto; o roteador real tem 9 classes por regex | `detect_query_intent` (`query_engine.py:225`) nunca chamado; `detect_query_type` (`:247`) chamado em `:700` |
| 3 | Fast/Slow path dual-stream é código morto | `memory/trg_memory.py:488` e `:549` só definidos, nunca chamados |
| 4 | Arestas `ENTITY` não existem no grafo construído | Única instância de `LinkType.ENTITY` em `trg_memory.py:667` (dentro do slow path morto); o builder grava `SEMANTIC/SAME_ENTITY` (`memory/memory_builder.py:753-755`) |
| 5 | Arestas causais vêm de regras sintáticas (troca de falante, `?`), não de LLM | `memory_builder.py:671-713` (`RESPONSE_TO`), `:839` (`ANSWERED_BY`) |
| 6 | Mapa fixo de tópicos → sessões do LoCoMo | `query_engine.py:174-180` (`'may': [1], 'july': [6,7,8,10]...`) — vazamento de benchmark |
| 7 | Uso do rótulo oracle `category == 5` no juiz, formatter e métricas | `memory/llm_judge.py:83`, `memory/answer_formatter.py:612,768`, `utils/utils.py:217` — contradiz "no oracle labels" (Ap. C.2) |
| 8 | Heurísticas regex específicas para o LongMemEval | `test_longmemeval_chunked.py:127,223` ("CRITICAL for reaching 80% accuracy") |

### 4.2 Nem o código morto segue o artigo
`_probabilistic_beam_search` (`query_engine.py:1002-1097`):
- Não usa `exp(...)` da Eq. 5 (soma linear).
- Não aplica o decaimento `γ` (`score_u·γ + s_uv`); usa `score + p`.
- Não é beam por profundidade (é best-first com heap global, sem `MaxDepth`).
- Pesos `w_Tq` (0.05–0.7) e `λ1=0.6` ≠ Tabela 6 (`w` 0.5–6.0, `λ1=1.0`).
- **Bug:** o corte do beam faz `heappop` num heap de scores negativos → remove o **melhor** candidato, não o pior.
- `get_neighbors` percorre arestas nos dois sentidos (inclusive causais) → a direção causal se perde.

### 4.3 O que ainda tem valor
| Item | Uso |
|---|---|
| Padrão `NetworkX MultiDiGraph` + dicts `nodes/links` + índice `node→links` (`memory/graph_db.py`) | **Reescrever no mesmo estilo** (~150 linhas), sem copiar |
| Configuração `LLM_API_KEY` / `LLM_MODEL` / `LLM_BASE_URL` (cliente OpenAI-compatível) | **Manter a convenção**: o mesmo código serve modelo local e provedores remotos |
| `detect_query_intent` (keywords WHY/WHEN/ENTITY) | Ponto de partida do router determinístico |
| LoCoMo (`data/locomo10.json`) + `load_dataset.py` | Opcional, só para o *sanity check* da Branch A |
| Prompt do juiz C.3 (do artigo, não o código) | Reescrever **sem** atalho de categoria |
| `memory/temporal_parser.py` (regex + timedelta, específico do LoCoMo) | Substituir por `dateparser` (determinístico, justificável) |
| Resto (answer_formatter, best_of_n, nós episode/session/narrative, script do LongMemEval, A-MEM em `utils/memory_layer.py`) | **Descartar** |

---

## 5. Decisão: reaproveitar vs reescrever → **REESCREVER DO ZERO** (fiel ao artigo)

Implementar o MAGMA *a partir do artigo* (Eqs. 3–7, Algoritmos 1–3, Tabela 6, Prompts C.1–C.3) num pacote novo e pequeno, sem importar código do repositório oficial.

**Justificativas**
1. **Fidelidade:** os componentes centrais do artigo (beam search, router de 3 intenções, dual-stream, arestas de entidade e causais por LLM) não estão ativos no código oficial (§4.1). Estendê-lo significaria estender outra arquitetura, não o MAGMA descrito.
2. **Validade experimental:** o código oficial usa rótulos oracle e mapas fixos do LoCoMo (§4.1 #6–#8). Uma comparação Original × Narrativa feita sobre ele ficaria contaminada.
3. **Variável controlada:** as duas branches compartilham exatamente o mesmo núcleo; a única variável independente é a extensão narrativa.
4. **Custo:** o núcleo fiel cabe em ~800–1200 linhas, contra ~14k linhas de heurísticas para entender e consertar.
5. **Licença:** o MIT permite reaproveitar ideias e estrutura; o repositório oficial é citado como referência, com as divergências documentadas (§4).

**Risco e mitigação:** o artigo é omisso em vários pontos (§1.8). Cada lacuna recebe valor escolhido + justificativa (§6.4).

---

## 6. Branch A (`magma-original`): reimplementação fiel

### 6.1 Princípios
- Cada módulo mapeia para uma equação/algoritmo e cita a seção no docstring (ex.: `# Eq. 5, §3.3`).
- **Sem rótulos oracle.** Nenhum código recebe categoria ou gabarito da pergunta.
- **Determinismo:** `temperature=0`, `seed` fixo, cache em disco das chamadas LLM (chave = hash do prompt + modelo).
- **Backend em memória** (NetworkX + NumPy), permitido pelo §3.5 do artigo.
- **Janela de contexto do LLM explícita:** o Ollama carrega o modelo com uma janela padrão (`num_ctx`) bem menor que a suportada pelo modelo e, se o prompt passar dela, descarta o início do texto, avisando só no log do servidor. `llm.py` sempre envia `num_ctx` (ex.: 32k) e acusa possível truncamento quando `prompt_eval_count` se aproxima do limite. Vale para todas as chamadas e todas as arquiteturas, com atenção especial ao baseline Full Context.
- Todos os hiperparâmetros num único `config.py` (espelho da Tabela 6) → ablações por flag.

### 6.2 Layout do pacote
```
magma/
  config.py      # Tabela 6 + valores escolhidos onde o artigo é omisso (§6.4)
  llm.py         # cliente OpenAI-compatível (LLM_API_KEY/LLM_MODEL/LLM_BASE_URL), cache em disco,
                 # contabiliza prompt/completion tokens e latência por chamada; grava prompt e resposta crus
                 # de toda chamada em runs/<run_id>/llm_calls.jsonl (D6)
  embed.py       # sentence-transformers (em processo, sem HTTP) carregando o modelo da D8: Qwen3-Embedding-0.6B
                 # (prompt de instrução só na query, vetores normalizados); all-MiniLM-L6-v2 só no sanity de fidelidade.
                 # O limite de corte é o `max_seq_length` do MODELO (MiniLM: 256 word-pieces; Qwen3: janela de 32k),
                 # não da biblioteca: loga o valor ao carregar e avisa quando um texto passa dele (o corte é silencioso)
  schema.py      # registro declarativo de tipos de nó e aresta + perfis (D9); perfil `magma` = Event, Entity; TEMP, CAUSAL, SEM, ENT
  graph.py       # núcleo agnóstico a tipo: Node{id,type,text,τ,v,attrs,visible_to}, Edge{src,dst,type,weight,attrs},
                 # tipos validados contra o perfil ativo do schema.py; NetworkX MultiDiGraph; persistência em JSON
                 # (save/load), como o código oficial (`graph_db.py:666-706`)
  index.py       # índice vetorial exato (D7) + índice esparso (BM25 via rank_bm25)
  temporal.py    # tagger temporal: dateparser com RELATIVE_BASE = timestamp do evento/sessão
  router.py      # Stage 1: intenção T_q∈{WHY,WHEN,ENTITY} + janela [τs,τe] + v_q + q_key
  ingest.py      # Alg.2 Fast Path (sem LLM) + Alg.3 Slow Path (fila + worker; LLM) + prompt C.1
  retrieve.py    # Stage 2 RRF (Eq.4), Stage 3 beam search (Alg.1, Eq.5-6), Stage 4 linearização (Eq.7) + budgeting
  qa.py          # prompt C.2 com instrução dinâmica gerada pelo router
  memory.py      # fachada: MagmaMemory.add_turn() / consolidate() / query()
bench/           # harness, métricas, juiz C.3, estatística (compartilhado com a Branch B)
baselines/       # full_context.py, naive_rag.py
tests/           # testes unitários de RRF, beam search e linearização com grafo de brinquedo
```

### 6.3 Fluxos
- **`add_turn(speaker, text, timestamp)`** — Fast Path (Alg. 2): cria `EventNode` (1 turno = 1 evento), aresta `TEMP` com o último nó, embedding + BM25, arestas `SEM` (cos > θ_sim, top-5 vizinhos), enfileira o id. **Nenhuma chamada LLM.**
- **`consolidate()`** — Slow Path (Alg. 3): para cada id da fila → prompt C.1 (entidades, tópico, fatos, datas, resumo) → cria/une `EntityNode` + arestas `ENT`; resolve `dates_mentioned` via `temporal.py` e grava em `A_i`; vizinhança 2-hop → LLM infere arestas `CAUSAL` com confiança ≥ δ.
  - Roda como **thread de fundo real** (fiel ao artigo), mas o benchmark chama `consolidate(drain=True)` antes de cada sonda → resultado determinístico.
- **`query(q)`** — Stage 1→4 (Alg. 1): router → RRF(vec, key, time) top-K âncoras → beam search com `score_v = score_u·γ + exp(λ1·w_Tqᵀ·1_r + λ2·sim)` → ordenação (WHEN: τ; WHY: sort topológico em `CAUSAL`; ENTITY: score) → `<t:τ> conteúdo <ref:id>` → nós de baixa saliência viram `"...N intermediate events..."` até caber no orçamento de tokens → prompt C.2.

### 6.4 Decisões onde o artigo é omisso
| Lacuna | Escolha | Justificativa |
|---|---|---|
| O que é um "evento" | 1 turno = 1 `EventNode` | Mais simples; igual ao uso em LoCoMo |
| C.1 usa LLM, mas o fast path não pode | C.1 roda no **slow path** | Respeita "no blocking LLM" do fast path |
| Classificador "leve" de intenção | Keywords + fallback por similaridade com protótipos (embedding) | Determinístico, custo zero, acurácia mensurável (D2) |
| Instrução dinâmica C.2 (4 categorias) × router (3 intenções) | WHEN→Temporal, WHY→Multi-hop/Inferência, ENTITY→Single-hop; "both/all/compare" → Multi-hop | Sem oracle; uma única fonte de roteamento |
| `w_phrase` | Peso da aresta `SEM` (similaridade entre trechos) | Único tipo de aresta sem peso declarado |
| Matriz `w_Tq` | Pontos dentro dos intervalos da Tab. 6 (ex.: WHY: causal 5.0, temp 1.5, ent 2.5, sem 2.5; WHEN: temp 4.0, causal 3.0, ent 2.5, sem 2.5; ENTITY: ent 6.0, causal 3.0, sem 2.5, temp 0.5) | Respeita os intervalos publicados |
| λ2 | 0.5 (meio do intervalo 0.3–0.7) | — |
| γ (decay) | 0.9 | Não declarado |
| BeamWidth | 10 | Valor do código-referência |
| δ (limiar causal) | 0.5 sobre a confiança emitida pelo LLM | Não declarado |
| θ_sim (aresta SEM) | 0.6, no máximo 5 vizinhos por nó | Evita a teia hiperdensa (~17 arestas/nó) do código oficial |
| Drop threshold 0.15 | Descarta vizinho com `sim(n_j,q) < 0.15` | Interpretação literal |
| Orçamento de contexto | ~2.5k tokens | Na ordem dos 3.37k tokens/query da Tab. 3 |
| Async | Thread real + `drain` antes das sondas | Fidelidade + reprodutibilidade |

### 6.5 Validação da Branch A (não é o benchmark principal)
- Testes unitários com grafo de brinquedo (RRF, beam, ordenação topológica).
- *Sanity run* num subconjunto pequeno do LoCoMo (~50 perguntas) para mostrar que o pipeline funciona e reproduzir as **quatro ablações da Tab. 4** (sem adaptive policy, sem causal, sem temporal, sem entity), todas como flags do `config.py`, além de medir build time, tokens/query e latência (Tab. 3). Sem o router adaptativo não há como validar a principal alegação de latência e acurácia do artigo. **Não** é objetivo bater 0.700.

---

## 7. Branch B (`magma-narrative`): extensão mínima para RF1–RF3

Branch B = Branch A **intacta** + pacote `narrative/`. O runner do benchmark na Branch B executa todas as arquiteturas (original, narrativa, baselines) com o mesmo LLM, embedding e prompts de agente → a única variável independente é o módulo de memória.

### 7.1 Ideia central (RF1): o *ato de fala* é fato; o *conteúdo* é alegação
- "B disse X para A e C no turno 12" é um **fato** (Event observável por quem estava presente).
- "X" é uma **alegação** (`Claim`) com status epistêmico próprio, que pode ser falsa.
- Isso resolve o *graph poisoning* já com 3 tipos de nó: mentiras ficam no grafo, mas nunca viram fato. Mais primitivos são possíveis sem refatorar (perfis da D9).

### 7.2 Esquema (perfil `narrative-min`, D9)
Os tipos abaixo são declarados em `narrative/schema.py`, que estende o perfil `magma` da Branch A. O `Event` funciona como **âncora** do turno; a `Claim` é uma **faceta** pendurada nele pela aresta de ancoragem `STATED` (D9).

**Nós**
| Tipo | Campos novos | Origem |
|---|---|---|
| `EventNode` (do artigo) | `actor`, `visible_to: set[agent] \| {"*"}`, `turn`, `action_text` (parte observável: gesto, narração), `speech_text` (fala); o conteúdo `c_i` continua sendo o turno inteiro | turno (separação determinística no fast path, D9) |
| `ClaimNode` (novo) | `statement`, `speaker`, `audience` (= `visible_to`), `about: [entidades]`, `status ∈ {UNVERIFIED, SUPPORTED, CONTESTED, REFUTED, RETRACTED}` | extração |
| `EntityNode` (do artigo) | `kind ∈ {agent, object, place}` | extração |

**Arestas** (8 nas 4 camadas do artigo + 1 de ancoragem; 9 tipos)
| Camada | Aresta | Semântica |
|---|---|---|
| Ancoragem (`anchor=True`) | `STATED` | evento de fala → claim que ele enuncia. Sem camada nem peso: o beam expande âncora e facetas juntas, sem custo de salto. O autor fica no atributo `speaker` |
| Temporal | `NEXT` (artigo) | cronologia |
| Temporal | `SUPERSEDES` | versão nova → versão anterior: claim do mesmo autor (anterior fica `RETRACTED`) ou fato atualizado (anterior ganha `superseded_by`). Obrigatória (D9) |
| Semântica | `SIM` (artigo) | cos > θ_sim |
| Entidade | `MENTIONS` (artigo) | evento/claim → entidade; inclui evento → ator, criada pelo fast path |
| Causal | `CAUSES` (artigo) | inferida no slow path |
| Causal | `INVALIDATES` | **fato** (`action_text` do evento) → claim que ele refuta |
| Causal | `SUPPORTS` | fato → claim que ele confirma |
| Causal | `CONTRADICTS` | claim ↔ claim incompatíveis de autores diferentes |

### 7.3 Escrita
- **Fast path (sem LLM):** igual à Branch A, mais `visible_to` vindo do **metadado de cena** (`present: [...]` do turno; ação privada → só o ator). Visibilidade **não é adivinhada por LLM**: num motor narrativo o orquestrador sabe quem está na cena. Isso torna o RF2 determinístico e auditável.
  - Separa o turno em `action_text` e `speech_text` por padrão de texto (`*ação* "fala"`, §7.5) e cria a aresta `MENTIONS` evento → ator. Sem LLM.
- **Slow path (LLM):** prompt C.1 **estendido** (uma chamada só) devolve também `claims[{statement, about, type}]`, extraídas **só do `speech_text`** → cria `ClaimNode`s ligados ao evento por `STATED`.
  - **Detecção de contradição e atualização (RF3):**
    1. Candidatos: claims **e** fatos anteriores com entidades em comum + top-k por similaridade.
    2. LLM em modo NLI `{contradicts | supports | unrelated, confidence}`. Quando a origem é um fato, a premissa é **só o `action_text`**: conteúdo de fala nunca vale como prova.
    3. Com o rótulo `contradicts`, uma regra determinística (`speaker` + `τ`, tabela da D9) escolhe a aresta: `INVALIDATES` (fato × claim), `CONTRADICTS` (autores diferentes) ou `SUPERSEDES` (mesmo autor ou fato × fato, versão mais nova).
    4. Com o rótulo `supports` e um fato como origem, cria `SUPPORTS`.
    5. Atualiza o `status` global.

### 7.4 Leitura (RF2): recuperação com escopo de observador
`query(q, viewer=agent_id)`:
1. **Filtro duro de visibilidade** antes do RRF (máscara nos índices vetorial e BM25) e durante o beam (vizinho invisível é ignorado). Vazamento na recuperação = 0 por construção.
2. **Status epistêmico relativo ao observador:** `status_for(viewer)` considera só as arestas `INVALIDATES/SUPPORTS/CONTRADICTS` cujo nó de origem é visível ao viewer. Quem não viu a prova continua achando a alegação "não verificada". É uma Teoria da Mente leve, sem grafo de crenças.
3. **Linearização em duas seções:**
   - `O que você presenciou`: o `action_text` do evento e o ato de fala ("B falou com {A, C}"), sem repetir o conteúdo.
   - `O que foi dito`: as claims, com autor, público e status visto pelo viewer, ex.: `<t:12> B disse a {A,C}: "Eu estava no laboratório" [REFUTADA por <ref:e35>] <ref:c12>`.
   - O status só é impresso quando é diferente de `UNVERIFIED`. Em benchmarks só de fala (LoCoMo, FANToM) quase tudo é claim não verificada; marcar todas gastaria tokens e induziria o LLM a desconfiar de tudo, o que prejudicaria a retenção (H2).
   - Para o mentiroso, as próprias claims aparecem como "você disse X a {…}" → base da consistência de engano (DCR).
4. **Router:** as mesmas 3 intenções; as arestas novas herdam o peso `w_Tq` da sua camada (D9), sem hiperparâmetro novo (ex.: `INVALIDATES`/`CONTRADICTS` são `CAUSAL`, peso alto em WHY; `SUPERSEDES` é `TEMP`, peso alto em WHEN; `MENTIONS` é `ENT`, peso alto em ENTITY). A aresta de ancoragem `STATED` não tem peso. Router idêntico nas duas branches (variável controlada).

### 7.5 Agentes
- `narrative/agents.py`: prompt de persona (identidade, relações, personalidade, objetivos, segredos) + memória com escopo + últimos K turnos visíveis → resposta no formato `*ação* "fala"`.
- **Orquestrador ao vivo faz parte do PoC:** é o "sistema de narrativa multiagente" do objetivo do projeto (agentes conversando entre si e com o usuário). Laço simples de turnos: escolhe quem fala, monta o prompt com a memória do viewer, gera a resposta e a ingere.
- **As métricas usam transcrições roteirizadas (replay)** → determinístico e barato (§8). O modo ao vivo serve de demonstração e de teste de integração.

### 7.6 Ablações (em tempo de consulta, sobre o mesmo grafo)
- `B − visibilidade` (ignora `visible_to`) → isola RF2.
- `B − status` (claims serializadas como fatos, sem status) → isola RF1.
- `B − contradição` (ignora `INVALIDATES/SUPPORTS/CONTRADICTS/SUPERSEDES`) → isola RF3.
- Espelha a Tab. 4 do artigo.

---

## 8. Benchmarks e avaliação

### 8.1 Trabalhos relacionados e benchmarks
| Trabalho | O que é | Relevância |
|---|---|---|
| **ReverieMem / KBF-QA** (arXiv 2606.25632) — *Staying In Character: Perspective-Bounded Memory for Book-Based Role-Playing Agents* | Memória em 3 camadas com **fatos marcados por visibilidade**; KBF-QA: 4.386 perguntas sobre 8 romances | ⚠️ **Correlato mais próximo do RF2.** Citar e diferenciar: eles tratam 1 personagem lendo um livro; este projeto trata diálogo multiagente com mentiras, separação fato/claim e atualização de status por contradição em multigrafo MAGMA |
| **Sotopia-ToM** (arXiv 2605.02307) | 160 cenários com assimetria de informação, 3–5 papéis, mede vazamento de privacidade | Referência para a métrica de vazamento (ELR). Abordagem por prompting (ToM-Coach), não por arquitetura de memória → bom contraste |
| **MemoryAgentBench / FactConsolidation** (arXiv 2507.05257) | Competência "conflict resolution": revisar memória diante de evidência contraditória; multi-hop ≤ 7% para todos os métodos | Benchmark externo para o **RF3**; dados em `ai-hyz/MemoryAgentBench` (Hugging Face) |
| ParliamentBench (arXiv 2607.28146), Mini-Mafia (arXiv 2509.23023) | Jogos de dedução social: engano sob assimetria; ParliamentBench tem *Deception Retention Rate* | Inspiração para a métrica DCR |
| TimeChara (2024), CharacterBench (2025), CHARM (arXiv 2609.01352) | *Knowledge boundary* de personagens | Correlatos do RF2 |
| **FANToM** (Kim et al., EMNLP 2023, arXiv 2310.15421) | 256 conversas multi-parte, ~10k perguntas; personagens **entram e saem** da conversa, criando assimetria de informação; perguntas de crença, "quem sabe X" (answerability) e acesso à informação | **Melhor encaixe público para o RF2:** a presença de cada personagem em cada fala define `visible_to` de forma determinística. Release oficial pelo GitHub dos autores; espelho no HF `TomTraining/TomDatasets` (cópia de terceiros, conferir) |
| **Diplomacy Deception** (Peskov et al., ACL 2020, "It Takes Two to Lie") | 17.289 mensagens de 12 partidas; cada mensagem rotulada pelo **emissor** (verdade/mentira) e pelo **receptor** (percebida como verdade/mentira) | Dados reais (não sintéticos) de mentira em diálogo multiagente; candidato para o RF1. HF `community-datasets/diplomacy_detection` (o card do HF tem título errado, "HateOffensive"). Não traz o estado do jogo → difícil gerar perguntas de fato |
| **LongMemEval — knowledge-update** (Wu et al., 2024) | Categoria de atualização de conhecimento; usada pelo próprio artigo do MAGMA (66.7%) | Alternativa ao FactConsolidation para o RF3, com comparação direta com o artigo; a variante *oracle* (só sessões de evidência) é bem menor |
| ExploreToM (Meta, arXiv 2412.12175; HF `facebook/ExploreToM`), OpenToM (arXiv 2402.06044), Hi-ToM (HF `umwyf/Hi-ToM_Dataset`) | ToM em histórias (crenças falsas, ordem superior); ExploreToM tem gerador próprio | Correlatos; narrativas passivas (não diálogo) |
| ConfAIde (2023) | Privacidade contextual, incluindo reuniões multi-parte | Clássico para citar no RF2 |

### 8.2 Proposta em discussão (D1): benchmarks públicos por requisito + diagnóstico próprio complementar

Princípio de validação: dados criados sob medida pelo próprio projeto enfraquecem a evidência. O ponto central não é "sintético × real" (o FANToM, por exemplo, foi gerado por LLM e validado por humanos), e sim **publicado, validado e usado por outros × feito sob medida**. Não existe benchmark público que cubra "memória multiagente com mentiras + visibilidade + contradição" ao mesmo tempo; essa lacuna é a própria motivação do trabalho. Proposta:

| Requisito | Benchmark | Papel |
|---|---|---|
| Fidelidade da Branch A + H2 (retenção) | **LoCoMo** (subconjunto de conversas) | Mesmo benchmark do artigo → comparabilidade direta |
| RF2 (visibilidade) | **FANToM** (subconjunto de conversas) | Evidência principal de assimetria de informação; arquiteturas oniscientes tendem a responder a verdade em vez da crença do personagem |
| RF3 (contradição/atualização) | **FactConsolidation** (menor tamanho de contexto) **ou** LongMemEval *knowledge-update* (oracle) | Evidência principal de resolução de conflito |
| RF1 (fato × declaração) | Diplomacy Deception (a avaliar) | Único dado real de mentira multiagente encontrado; falta definir a tarefa de avaliação |
| Integração RF1+RF2+RF3 | Conjunto diagnóstico próprio (Épsilon-4 + 2 cenários) | **Complementar**, rotulado como tal; gabarito validado por dois anotadores independentes, com concordância reportada |

Tamanho dos subconjuntos: definido depois de medir a vazão de inferência local no M0.

### 8.3 Formato do conjunto diagnóstico próprio (`bench/scenarios/*.json`)
Os benchmarks públicos são convertidos para este mesmo formato por adaptadores (`bench/adapters/`), para que o runner seja um só. Ex.: no FANToM, `present` vem das entradas e saídas de personagens.
```json
{
  "id": "epsilon4",
  "agents": [{"id": "B", "name": "...", "persona": "...", "secrets": ["s1"], "goals": ["..."]}],
  "turns": [
    {"t": 5, "time": "2187-03-02T21:40", "actor": "B", "present": ["B"], "text": "*B hides the chip inside the vent.*"},
    {"t": 12, "time": "2187-03-02T22:30", "actor": "B", "present": ["A", "B", "C"], "text": "\"I was in the lab all night.\""}
  ],
  "gold": {
    "facts":  [{"id": "f1", "turn": 5, "text": "B hid the chip in the vent", "witnesses": ["B"]}],
    "claims": [{"id": "c1", "turn": 12, "speaker": "B", "text": "B was in the lab all night", "truth": false}],
    "invalidations": [{"claim": "c1", "by_turn": 35}]
  },
  "probes": [
    {"id": "p1", "at_turn": 20, "viewer": "C", "type": "leakage",      "question": "Where is the chip?", "forbidden": ["f1"]},
    {"id": "p2", "at_turn": 20, "viewer": "B", "type": "deception",    "question": "Where were you at 22h?", "cover_story": "c1"},
    {"id": "p3", "at_turn": 40, "viewer": "D", "type": "invalidation", "question": "Which of B's statements were disproven, and by what?", "gold_claim": "c1", "gold_evidence_turn": 35},
    {"id": "p4", "at_turn": 40, "viewer": "*", "type": "qa", "intent_gold": "WHEN", "question": "...", "answer": "..."}
  ]
}
```
Execução: ingestão incremental turno a turno; ao atingir `at_turn`, `drain` da consolidação → roda as sondas daquele ponto. Respeita o caráter variante no tempo de `G_t`.

### 8.4 Métricas (variáveis dependentes)
| Métrica | Definição | Como medir | RF |
|---|---|---|---|
| **RLR** — *Retrieval Leakage Rate* | % de nós recuperados fora do `visible_to` do viewer | **Determinístico** (gabarito de testemunhas) | RF2 |
| **ELR** — *Epistemic Leakage Rate* | % de respostas que revelam fato proibido | Juiz LLM + checagem lexical das keywords do segredo | RF2 |
| **DCR** — *Deception Consistency Rate* | % de sondas em que o mentiroso mantém a versão que contou | Juiz LLM com `cover_story` | RF1 |
| **CIA** — *Causal Invalidation Accuracy* | Investigador identifica a claim refutada **e** a evidência certa | Juiz + match de `<ref:id>` | RF3 |
| **Contradiction F1** | P/R/F1 das arestas `INVALIDATES/CONTRADICTS` vs gabarito | **Determinístico**, sem juiz | RF3 |
| **QA Judge** | Prompt C.3 do artigo em sondas factuais/temporais/causais | Juiz | H2 (retenção) |
| **Router accuracy** | intenção prevista × `intent_gold` | Determinístico | D2 |
| **Tokens/query, latência/query, tempo de build, nós/arestas** | contadores do `llm.py` (`prompt_eval_count`, `eval_count` no Ollama) | Determinístico | RF4 / H3 |

Nos benchmarks públicos, reportar **primeiro as métricas oficiais de cada um** (ex.: acurácia por tipo de pergunta no FANToM) e só depois as métricas acima que forem computáveis (ex.: RLR no FANToM).

**Estatística:** média ± IC 95% (bootstrap); teste pareado (McNemar para métricas binárias, permutação para contínuas) Narrativa × cada controle.

### 8.5 Grupos (controle × experimental)
| Grupo | Memória | Papel |
|---|---|---|
| C1 Full Context | transcrição inteira no prompt | baseline tradicional (também usado na Tab. 1 do artigo). **Definir `num_ctx` explícito no Ollama**: o padrão é bem menor que a janela do modelo e trunca o prompt sem avisar. Registrar quando a transcrição não couber |
| C2 Naive RAG de terceiros | LlamaIndex (ou LangChain) top-k, mesmo embedding/LLM | **Origem: requisito metodológico do projeto (§2), não do artigo.** O artigo MAGMA compara com Full Context, A-MEM, Nemori e MemoryOS, sem RAG ingênuo; o código oficial também não tem esse baseline. Representa a "memória tradicional" mais comum (busca vetorial top-k, sem grafo). **Chunk ≤ janela do embedding** (ex.: 1 turno por chunk): o chunk padrão do LlamaIndex (1024 tokens) passa da janela do MiniLM, e o excesso seria cortado em silêncio, prejudicando injustamente o baseline. Com o embedding da D8 (32k) o risco some, mas o chunk continua sendo 1 turno, para igualar a granularidade das outras arquiteturas |
| C3 MAGMA original | Branch A | arquitetura base |
| E MAGMA Narrativa | Branch B | proposta |
| E−x | ablações §7.6 | contribuição de cada RF |

Todos usam o **mesmo prompt de persona, mesmo LLM e mesmas sondas**; só muda a memória.

---

## 9. Decisões pendentes

- [ ] **D1 — Benchmark.** Em discussão (§8.2): benchmarks públicos por requisito (LoCoMo → fidelidade/H2; FANToM → RF2; FactConsolidation ou LongMemEval-KU → RF3; Diplomacy → RF1, a avaliar) + conjunto próprio só como diagnóstico complementar. Em aberto: tarefa de avaliação do RF1 e tamanho dos subconjuntos.
- [x] **D7 — Índice vetorial.** ✅ **Decidido:** FAISS `IndexFlatIP`, k-NN exato, local, igual para todas as arquiteturas.
  - **Terminologia:** "busca exata" aqui = **k-NN exato** sobre embeddings (compara a query com *todos* os vetores e devolve os k mais próximos de verdade), em oposição ao **k-NN aproximado** (HNSW/IVF: atalhos que podem perder vizinhos verdadeiros). **As duas são busca semântica** ("guerra" encontra "conflito"). Não confundir com busca **lexical** (BM25, palavra igual), que é outro canal do MAGMA. FAISS é biblioteca de busca por similaridade em memória, não é banco de dados nem "RAG" (RAG é o padrão recuperar → gerar).
  - **Os 3 canais de recuperação do artigo (Stage 2, Eq. 4):** semântico (embedding denso + índice vetorial), lexical (keywords esparsas) e temporal (janela `[τs, τe]`), fundidos por RRF. Cada canal cobre a falha do outro: o semântico acha sinônimos e paráfrases; o lexical acha nomes próprios e termos raros ("Épsilon-4", "chip").
  - **Onde o índice vetorial entra no artigo:** Eq. 3 (`v_i` indexado no vector DB, citando FAISS); Alg. 2 (`VDB.ADD` no fast path); Stage 2 (`V.SEARCH(v_q)`, top-K = 20, canal semântico do RRF); grafo semântico (`cos(v_i,v_j) > θ_sim` exige achar os vizinhos mais próximos de cada nó). Stage 3 (`sim(v_j, v_q)` no beam) só lê o vetor do nó pelo id, sem busca.
  - **Onde entra no código oficial:** `memory_builder.py:645` (`create_semantic_links`, k-NN para arestas `SEMANTIC`); `trg_memory.py:262` (`query`, âncoras, chamado em `query_engine.py:738`); `add_vector` na ingestão (`memory_builder.py:350`; `:400` para os resumos de sessão, que não estão no artigo).
  - **Onde entra nesta implementação:** (1) canal semântico das âncoras (Stage 2); (2) arestas `SEM` no fast path; (3) Branch B: candidatos para a detecção de contradição (claims parecidas); (4) baseline C2. O beam só lê vetores por id.
  - **Modo usado:** o artigo cita FAISS (Johnson et al., 2019), mas **não informa o tipo de índice**. O código oficial usa por padrão o **índice exato**: `FAISSVectorDB(index_type="flat")` → `IndexFlatL2` (`memory/vector_db.py:136,158`); a factory `backend="auto"` escolhe FAISS se instalado e, se não, NumPy de força bruta, também exato. IVF/HNSW existem no código, mas não são o padrão. Como o MiniLM gera vetores normalizados, ordenar por distância L2 dá o mesmo resultado que ordenar por cosseno.
  - **Recomendação:** FAISS `IndexFlatIP` (produto interno sobre vetores normalizados = cosseno; mesma ordenação do `IndexFlatL2` do código, mas devolve o cosseno direto, que é o que θ_sim e `sim(·)` usam), k-NN exato, igual para todas as arquiteturas; o baseline C2 pluga o mesmo índice no LlamaIndex.
  - **Alternativas comparadas:** k-NN aproximado (FAISS IVF/HNSW), bancos vetoriais locais (pgvector, Qdrant, Chroma, LanceDB, Milvus Lite) e bancos remotos. Motivos para descartar abaixo.
  - **Por quê:** (1) na escala do projeto (centenas a poucos milhares de nós) a busca exata leva menos de 1 ms, e índices aproximados (HNSW/IVF) só compensam a partir de ~10⁵–10⁶ vetores; (2) busca exata elimina o erro de aproximação como variável de confusão; (3) um banco remoto colocaria latência de rede dentro da métrica de latência (RF4/H3); (4) zero infraestrutura.
  - **pgvector:** a fama de "fraco" vem de vazão e recall em grande escala, o que é irrelevante aqui; o custo real seria subir um servidor Postgres. Qdrant, Chroma, LanceDB e Milvus Lite também rodam localmente, mas pelo mesmo motivo não acrescentam nada nesta escala.
  - **Esclarecimento:** "RAG de terceiros" no plano significa **implementação** de terceiros (framework), não banco remoto.
  - **Quando faria sentido um banco vetorial de verdade:** para estudar escalabilidade (10⁵+ nós) ou memória persistente entre sessões → trabalhos futuros.
- [x] **D8 — Modelo de embedding.** ✅ **Decidido:** `Qwen3-Embedding-0.6B` principal, igual para todas as arquiteturas; `all-MiniLM-L6-v2` só na checagem de fidelidade da Branch A. Transforma cada texto (evento, claim, pergunta) no vetor que o índice da D7 compara.
  - **Contexto:** o artigo usa por padrão `all-MiniLM-L6-v2` (Tab. 6), com `text-embedding-3-small` (API) como opcional → o próprio artigo trata o embedding como peça trocável. O MiniLM é de 2021 e só inglês; projetos atuais usam modelos mais novos.
  - **Opções:**
    | Modelo | Ano | Tamanho | Idiomas | A favor | Contra |
    |---|---|---|---|---|---|
    | `all-MiniLM-L6-v2` | 2021 | 22M params, 384d | inglês | Default do artigo → fidelidade e comparabilidade; muito rápido (até em CPU) | Antigo; qualidade inferior aos atuais; só inglês; janela curta (256 tokens, suficiente para turnos de diálogo) |
    | `text-embedding-3-small` | 2024 | API | multi | Opcional do artigo | Remoto e pago; latência de rede na métrica; não reprodutível offline |
    | `Qwen3-Embedding-0.6B` | 2025 | 0.6B | 100+ | Melhor qualidade por VRAM entre os locais (~1.5 GB); Ollama; Apache 2.0 | Mais lento que o MiniLM (ainda rápido em GPU); usa instrução na query |
    | `BGE-M3` | 2024 | 568M, 1024d | 100+ | Consolidado e muito citado; denso + esparso num modelo só | Maior; o modo esparso duplicaria o canal BM25 |
    | `nomic-embed-text` | 2024 | ~137M | inglês | Leve, nativo no Ollama | Abaixo do Qwen3 em qualidade |
    | BERT "puro" (ex.: `bert-base`) | 2018 | 110M | — | — | **Não serve:** BERT sem ajuste contrastivo gera vetores ruins para similaridade. Os modelos acima já *são* encoders da família BERT ajustados para similaridade (abordagem Sentence-BERT, Reimers & Gurevych, 2019) |
  - **Janela de 256 tokens não limita o tamanho da conversa:** ela vale **por texto embutido**. O MAGMA gera um vetor por evento (1 turno = 1 nó), então uma conversa de 10 mil turnos vira 10 mil vetores curtos; o tamanho da conversa afeta só o número de vetores, que a busca exata do FAISS absorve (10⁵ vetores × 384d ainda custam milissegundos). O risco real é um **único turno** passar de ~256 word-pieces (~190 palavras em inglês): o `sentence-transformers` **corta o excesso sem avisar** no `max_seq_length` do modelo carregado (o limite é do modelo, não da biblioteca). Turnos de LoCoMo, FANToM e FactConsolidation são curtos; respostas longas de assistente (ex.: LongMemEval) podem estourar.
    - **No PoC:** com a janela de 32k do modelo recomendado, o truncamento não ocorre na prática. A diluição em mensagens longas (abaixo) é limitação conhecida, que o artigo também não trata, e fica para depois do PoC (§12). Mantém-se 1 turno = 1 evento.
    - **Dois problemas distintos:** (a) **truncamento**: o texto passa da janela e o excesso é descartado (só acontece com janela curta; com 32k, um texto de 2.000 palavras ≈ 2.700 tokens cabe inteiro); (b) **diluição**: qualquer embedding comprime o texto inteiro num **único vetor de tamanho fixo**, seja ele uma frase ou um relatório. Num relatório com cinco assuntos, o vetor representa a "média" deles, e uma pergunta sobre um detalhe ("onde o chip foi escondido?") casa forte com o parágrafo certo, mas fraco com o relatório inteiro. A janela longa resolve (a), mas não (b).
    - **Melhoria futura (§12): segmentação de eventos.** Mensagem longa (ex.: relatório de um agente) viraria vários eventos de até ~512 tokens, cortados em fronteira de parágrafo/frase e encadeados por `TEMP`. Por que segmentar mesmo com janela longa: (1) diluição: unidades menores dão buscas mais precisas (Chen et al., 2024, *Dense X Retrieval: What Retrieval Granularity Should We Use?*, mostra ganho ao indexar proposições em vez de passagens); (2) orçamento de contexto: um nó de 2.700 tokens sozinho estoura o orçamento de ~2.5k da linearização e não deixa espaço para outras evidências; (3) precisão do grafo: arestas causais e de entidade entre eventos atômicos dizem *qual* fato causa *qual*, enquanto entre dois relatórios inteiros ficam vagas. O limite de ~512 tokens por evento é parâmetro de projeto (`config.py`) e pode virar ablação.
  - **Recomendação (revisada):** `Qwen3-Embedding-0.6B` como embedding **principal e igual para todas as arquiteturas**; `all-MiniLM-L6-v2` só num *sanity run* opcional da Branch A, como checagem de fidelidade à configuração original.
  - **Por quê:** (1) robustez a mensagens longas, comuns em narrativa multiagente real (janela de 32k contra 256); (2) qualidade bem superior e multilíngue; (3) o argumento "comparabilidade com os números do artigo" é fraco: o LLM já é outro (local, não gpt-4o-mini), então os números absolutos não seriam comparáveis de qualquer forma; o que se compara é a tendência das ablações e o ranking entre arquiteturas; (4) o embedding continua variável controlada (igual para todos); (5) local, Apache 2.0, ~1.5 GB de VRAM; (6) trocar o modelo é uma linha no `config.py`.
  - **Contras aceitos:** mais lento que o MiniLM (ainda rápido em GPU); as queries usam instrução no formato `Instruct: ...\nQuery: ...` (documentar o template usado); vetores de 1024d (índice maior, irrelevante nesta escala).
  - **Alternativa mais próxima:** `BGE-M3` (janela de 8k, artigo próprio publicado em 2024, muito citado). Fica como segunda opção se o Qwen3-Embedding der problema de instalação ou de desempenho.
  - **Fonte da comparação de modelos (blogs, números não comparáveis entre si):** https://zilliz.com/blog/choose-embedding-model-rag-2026 · https://d-central.tech/local-embedding-models/ · https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/
- [ ] **D2 — Roteamento.** Recomendação: router **determinístico** (keywords WHY/WHEN/ENTITY + fallback por protótipos de embedding), igual nas duas branches; router por LLM só como ablação opcional. Alternativa em avaliação (termo a confirmar): **JEPA** (Joint-Embedding Predictive Architecture), cujo análogo determinístico é o fallback por protótipos de embedding, ou **GEPA** (otimizador de prompts genético-Pareto, DSPy), que só faz sentido com router por LLM e exige muitas chamadas.
- [ ] **D3 — Modelos.** Recomendação: um único LLM local para extração, agentes e QA (candidatos: `qwen3:14b`, ~9 GB em Q4, ou `gpt-oss:20b`, ~13 GB; escolher conforme a VRAM disponível), `temperature=0`, `seed=42`. **Juiz de família diferente** (evita viés de autopreferência), ex.: Gemini Flash via API ou outro modelo local. Embedding: definido na D8. Verificar disponibilidade dos modelos no Ollama no setup.
- [x] **D4 — Baselines (sistemas de controle).** ✅ **Decidido:** C1 Full Context + C2 RAG ingênuo via LlamaIndex + C3 MAGMA original. Baseline ≠ benchmark: benchmark é o conjunto de dados + métrica (D1); baseline é outro sistema de memória rodando **no mesmo benchmark**, com o mesmo LLM.
  - **Como o artigo fez (Ap. D):** Full Context foi implementado pelos autores; A-MEM, Nemori e MemoryOS rodaram a partir das **implementações oficiais** de cada um, com hiperparâmetros padrão; todos com o mesmo LLM (gpt-4o-mini) para isolar o efeito da memória.
  - **Por que não dá para reaproveitar os números publicados:** o nosso LLM é outro (local). Comparar o nosso sistema com números obtidos com gpt-4o-mini misturaria o efeito do LLM com o da memória. Os números publicados só podem aparecer como contexto, com essa ressalva.
  - **Recomendação para o PoC:** C1 Full Context (feito por nós, como no artigo) + C2 RAG ingênuo de terceiros (LlamaIndex `VectorStoreIndex`, só indexação e recuperação, mesmo embedding da D8 e mesmo prompt de resposta) + C3 MAGMA original (Branch A).
  - **Por quê:** (1) C1 e C2 são baratos: não fazem extração por LLM na ingestão, só a chamada de resposta, e o runner é o mesmo; (2) C2 atende o requisito metodológico de comparar com uma implementação de terceiros (§2); (3) no FANToM, o próprio benchmark avalia LLMs lendo a conversa inteira, então C1 reproduz a configuração oficial dele.
  - **Alternativas descartadas no PoC:** rodar A-MEM, Nemori e MemoryOS oficiais com o nosso LLM local. É o mais fiel ao artigo, mas cada um exige integração própria (dependências, adaptação de formato, troca de backend de LLM) → §12. Mem0 OSS: mesma razão.
- [x] **D6 — Camada de acesso a LLM.** ✅ **Decidido:** **sem framework de LLM no núcleo**. Um `magma/llm.py` próprio (~80 linhas) sobre o SDK `openai` (cliente HTTP fino e tipado) apontado para qualquer endpoint OpenAI-compatível (Ollama local por padrão; provedor remoto só para o juiz), com `pydantic` (só validação de dados) para checar o JSON das extrações. Interface: `chat(messages, schema=None, tag=...) -> {text, parsed, prompt_tokens, completion_tokens, latency_s, cached}`.
  - **Por quê:** (1) o roteamento, a recuperação e a consolidação *são* a contribuição, e não podem ficar dentro de abstrações de terceiros; (2) tokens e latência são variáveis dependentes (RF4/H3), e overhead ou retries escondidos de framework contaminariam a medição; (3) todo prompt citado no relatório é exatamente a string enviada, auditável no `llm_calls.jsonl`; (4) frameworks de LLM mudam de API com frequência → risco de reprodutibilidade; (5) os "agentes" aqui são prompts com contexto recuperado, sem tool calling, e o orquestrador é um laço simples de turnos. LangGraph modela grafo de *controle de fluxo*, não grafo de *memória*.
  - **Descartados:** LangChain/LangGraph (abstração pesada, templates e parsers ocultos); PydanticAI (mais leve e tipado, mas ainda é um laço de agente que injeta instruções/ferramentas no prompt para saída estruturada); LiteLLM (desnecessário: endpoints OpenAI-compatíveis já cobrem local e remoto).
  - **Exceção:** o baseline C2 usa LlamaIndex **só para indexar e recuperar** (`retriever.retrieve`). A resposta usa o mesmo prompt de persona/QA das outras arquiteturas, para que a única variável seja a memória.
  - **Provedor remoto:** se usar um gateway que roteia entre vários upstreams, fixar provedor e desligar fallback (o mesmo id de modelo pode cair em backends diferentes → não determinismo). Registrar em cada run: id do modelo, provedor, versão do Ollama e digest do modelo local.
- [ ] **D5 — Idioma dos cenários.** Observação: todos os benchmarks públicos propostos na D1 são em inglês, então o conjunto diagnóstico próprio também deve ser, por consistência. Recomendação: **inglês** (modelos locais rendem mais, o embedding do artigo é inglês, comparável com a literatura). Alternativa: português + embedding multilíngue (`paraphrase-multilingual-MiniLM-L12-v2`).
- [ ] **D9 — Ontologia (tipos de nó e aresta) e extensibilidade.** Recomendação em discussão. Pergunta: quantos tipos de nó e de aresta o PoC usa, e como deixar o código pronto para testar outras ontologias sem refatorar? A ontologia dos rascunhos (§3.1: 5 primitivos, ~20 arestas) foi **inspiração**, não requisito: a proposta abaixo se justifica sozinha.
  - **Recomendação:** separar **mecanismo** de **ontologia**.
    1. O núcleo (`graph.py`, `retrieve.py`, linearização) não conhece nomes de tipo. Ele lê tudo de um **registro declarativo** (`schema.py`), em que cada tipo é um objeto Python.
    2. A ontologia vira um **perfil** escolhido no `config.py`:
       - `magma` (Branch A, igual ao artigo);
       - `narrative-min` (§7.2, o perfil medido no PoC).
    3. **Perfis enriquecidos** entram quando a análise de erros nos benchmarks mostrar uma relação que falta (ex.: perguntas erradas por não haver posse ou localização), e não a priori. A tabela de candidatos abaixo registra o que já foi considerado.
    4. A granularidade da ontologia passa a ser **variável experimental** decidida por benchmark, não por opinião.
  - **O que cada tipo declara:**
    - **Nó:** nome; descrição (entra no prompt de extração); se é âncora temporal ou faceta; se é embutido (embedding/BM25).
    - **Aresta:** nome; descrição; tipos de origem/destino permitidos; direcionada ou não; **camada** (`TEMP`, `CAUSAL`, `SEM`, `ENT`) **ou** `anchor=True` (aresta de ancoragem de faceta, sem camada); **produtor**; **efeito epistêmico** (`supports`, `refutes`, `contradicts`, `retracts`, `updates` ou nenhum); template de linearização.
    - **Produtores possíveis:**
      - `rule`: código no fast path, ex.: `NEXT`, `SIM`;
      - `llm_extract`: slow path, campo gerado no prompt de extração a partir da descrição;
      - `nli`: checagem de contradição;
      - `derived`: calculado na consulta, ex.: `status_for`.
    - **Consequência:** com `llm_extract` genérico, um tipo novo é só uma declaração; código novo só é preciso quando o produtor é uma regra própria.
  - **Por que os pesos não explodem:** o score do beam (Eq. 5) usa `w_Tq` **por camada** (4 valores por intenção, com intervalos publicados na Tab. 6). Cada tipo novo herda o peso da sua camada, com multiplicador opcional (padrão 1).
    - Pesos por tipo criariam 3 intenções × ~20 tipos ≈ 60 hiperparâmetros sem referência publicada.
    - No perfil `magma` o mapeamento tipo → camada é 1:1, então a fórmula fica idêntica à do artigo.
  - **Status por efeito:** `status_for(viewer)` (§7.4) lê o **efeito** declarado da aresta, não o nome dela. Uma aresta nova que refuta entra no cálculo de status sem mudar código.
  - **Definição de faceta:** nó que pertence a **exatamente um** `Event` âncora, ligado a ele por uma aresta com `anchor=True`, e que herda dele `τ` e `visible_to`. No `narrative-min` a única faceta é a `Claim`, e a aresta de ancoragem é `STATED` (evento de fala → claim que ele enuncia).
    - **Origem:** a estrutura vem da ideia central (§7.1: o ato de fala é fato, o conteúdo é alegação), e já estava na §7.2 original. O nome vem do *Dual-Facet Event* dos rascunhos.
    - **Autoria sem aresta extra:** o autor fica no atributo `speaker`, e o fast path liga o evento ao ator por uma aresta `ENT` determinística. Como a âncora traz as facetas sem custo de salto, "o que B disse" é **um** salto ponderado: `B ←ENT– evento`, com as claims vindo junto.
    - **Consequência:** uma proposição repetida em outra fala (ou repassada por outro agente) gera **outra** claim, ligada à anterior por `SIM` (`SUPPORTS` é só fato → claim). Um fato que refute a proposição é ligado por `INVALIDATES` a cada uma delas, já que todas são candidatas por similaridade; cada viewer vê o status da versão que ouviu.
    - **Alternativas descartadas:**
      - *Claim como proposição única, compartilhada por várias falas:* autor, público e status deixariam de caber num nó só, o que complica o RF2 sem ganho medível no PoC.
      - *Aresta estrutural própria (`HAS_FACET`) + `STATED` como agente → claim:* foi proposta para casar com a taxonomia dos rascunhos e retirada na revisão. A aresta agente → claim é redundante (o agente já chega às próprias claims em um salto, como acima) e acrescentaria um tipo extraído sem pergunta de benchmark que o meça.
  - **Separação fato × fala dentro do turno:** o texto cru de um turno mistura o observável (gesto, narração) com o conteúdo da fala. Se a detecção de contradição usasse o texto inteiro do evento como "fato", o conteúdo de uma mentira poderia refutar ou confirmar outras claims como se fosse prova: o *graph poisoning* voltaria pela porta do NLI.
    - **Regra:** o fast path separa o turno, sem LLM, em `action_text` (parte observável) e `speech_text` (fala).
      - No formato narrativo `*ação* "fala"` (§7.5) a separação é por padrão de texto.
      - Turnos de LoCoMo e FANToM são só fala.
      - Turnos de narração (`actor: narrator`) e fatos de benchmarks de atualização são só `action_text`.
    - **Usos:**
      - O vetor do evento continua sendo o do texto inteiro, como no artigo.
      - A premissa do NLI para `INVALIDATES`/`SUPPORTS` é **só** o `action_text`. As claims são extraídas só do `speech_text`.
      - Fala contra fala gera `CONTRADICTS`, nunca `INVALIDATES`.
    - **Na linearização:** o evento mostra o `action_text` e o ato de fala ("B falou com {A, C}"); o conteúdo aparece uma vez só, na claim.
  - **Atualização × contradição (`SUPERSEDES` obrigatória):** nem toda incompatibilidade é mentira. Quando o NLI responde `contradicts`, uma regra determinística (rótulo do NLI + `speaker` + `τ`) escolhe a aresta:
    | Novo × anterior | Aresta | Efeito no anterior |
    |---|---|---|
    | Fato × claim | `INVALIDATES` | claim `REFUTED` |
    | Claim × claim, autores diferentes | `CONTRADICTS` | claim `CONTESTED` |
    | Claim × claim, mesmo autor, mais nova | `SUPERSEDES` | claim `RETRACTED` (mudou a versão) |
    | Fato × fato, mais novo | `SUPERSEDES` | fato marcado `superseded_by` (o mundo mudou) |
    - **Por quê:** sem a regra, uma atualização legítima ("mudei para Seattle" depois de "moro em Boston") ficaria `CONTESTED`, e o QA não saberia qual versão vale. É exatamente o que FactConsolidation e LongMemEval-KU medem (D1). O mesmo vale para o mentiroso que muda de versão no diagnóstico próprio.
    - **Candidatos do NLI:** claims **e** fatos anteriores com entidades em comum + top-k por similaridade (fatos entram para detectar atualização).
  - **Regra das facetas na recuperação (vale para qualquer perfil):**
    - **Recomendação:** a faceta tem **vetor e BM25 próprios**, mas um acerto nela é **elevado à âncora**.
      1. **Indexação:** cada faceta é embutida e indexada (vetorial + BM25) com o próprio texto. Ela herda `τ` e `visible_to` do `Event` âncora, salvo exceção declarada (ex.: gesto privado visível só ao ator).
      2. **Stage 2 (RRF):** acertos em facetas são agregados por âncora. O rank da âncora é o melhor rank entre ela e suas facetas, e os top-K = 20 contam **âncoras únicas**.
      3. **Stage 3 (beam):** as facetas entram no grafo como nós comuns. Ganham score pelo próprio vetor (`sim(v_j, v_q)`), e suas arestas (`INVALIDATES` chegando, `CONTRADICTS` e `SUPERSEDES` entre claims) são percorridas normalmente. A aresta de ancoragem não tem custo de salto.
      4. **Stage 4 (linearização):** cada faceta é impressa **aninhada** na linha do seu evento, na seção "O que foi dito", com o status visto pelo viewer. O budgeting corta o evento com as facetas juntas, nunca a faceta solta.
    - **Por quê:**
      - **Precisão:** a claim extraída ("B passou a noite no laboratório") tem vetor próprio, sem a diluição do turno inteiro (o mesmo problema da D8).
      - **Comparação justa:** o conjunto de âncoras tem o mesmo tipo e o mesmo tamanho em qualquer perfil. A diferença medida entre perfis vem da ontologia, não de mudar o ponto de partida do beam.
      - **Canal temporal intacto:** o `τ` da âncora alimenta o canal temporal do RRF, como no artigo.
      - **Contexto completo:** o LLM nunca recebe uma faceta sem o evento a que ela pertence.
    - **Alternativas descartadas:**
      - *Facetas como âncoras independentes:* um mesmo turno ocuparia várias das 20 vagas do top-K, expulsando outros eventos, e repetiria o mesmo conteúdo no prompt. O número de âncoras também mudaria de um perfil para outro (confusão na comparação).
      - *Facetas sem embedding, alcançadas só por travessia:* não quebra, mas perde precisão.
        - A busca e a seleção de candidatas do NLI passam a usar o vetor do turno inteiro como substituto. Num turno com várias claims (ex.: álibi + acusação + comentário), o vetor mistura todas.
        - Por entidade em comum, o suspeito aparece em quase todas as claims, então a entidade sozinha não ordena as candidatas.
        - O custo do Caso 1 é pequeno: um embedding a mais por claim e a agregação por âncora no RRF.
        - → Descartada como padrão. Pode virar flag de ablação (`facet_vectors=False`) se for preciso medir o ganho.
  - **Por que o PoC mede o perfil mínimo primeiro:**
    1. Cada tipo extraído por LLM aumenta o JSON do slow path. Com modelo local isso traz mais falhas de parsing e mais tokens, e tokens e latência são variáveis dependentes (RF4/H3).
    2. Os benchmarks da D1 (LoCoMo, FANToM, FactConsolidation) só perguntam quem estava presente, o que foi dito e o que foi refutado ou atualizado. Arestas como `OWNS`, `LOCATED_IN` e `TARGETS` não têm pergunta que as teste, então o benchmark não diria se elas ajudam.
    3. Vários tipos candidatos viram atributo sem perda para os RFs. A tabela abaixo mostra cada um.
    4. É o menor perfil em que cada aresta tem produtor, consumidor e métrica.
  - **Catálogo de candidatos (rascunhos) → `narrative-min`:**
    | Candidato | No `narrative-min` | Se um perfil enriquecido precisar |
    |---|---|---|
    | `Agent` | `Entity{kind=agent}` | nó próprio |
    | `Entity` | `Entity` | `Entity` |
    | `Event` | `Event` (âncora) | `Event` (âncora) |
    | `PhysicalAction` | atributo `action_text` do `Event` | faceta própria (ancoragem), com vetor próprio |
    | `Claim` | `Claim` (faceta via `STATED`) | idem |
    | `NEXT_IN_TIME` | `NEXT` | `NEXT` |
    | `SIMILAR_TO` | `SIM` | `SIM` |
    | `STATED` (agente → claim) | atributo `speaker` + `ENT` evento → ator | aresta agente → claim |
    | `CAUSES` | `CAUSES` | `CAUSES` |
    | `INVALIDATES` | `INVALIDATES` | `INVALIDATES` |
    | `CONTRADICTS` | `CONTRADICTS` | `CONTRADICTS` |
    | `SUPERSEDES` | **obrigatória** (regra acima) | idem |
    | `PERFORMED` | atributo `actor` + `ENT` evento → ator | aresta |
    | `ADDRESSED_TO` | atributo `audience` (= `visible_to`) | aresta |
    | `BELIEVES {confidence}` | derivado: `status_for(viewer)` | aresta `derived` ou `llm_extract` |
    | `DENIES` | `Claim` negada + `CONTRADICTS` | aresta, efeito `contradicts` |
    | `MOTIVATES` | — | aresta `CAUSAL` |
    | `REPLIES_TO` | — | aresta `TEMP` |
    | `TARGETS`, `USES`, `LOCATED_IN`, `OWNS` | — | arestas `ENT` |
    | `THEMATICALLY_RELATED` | — | aresta `SEM` |
    | *(novo)* `MENTIONS` | ENT do artigo | idem |
    | *(novo)* `SUPPORTS` | efeito `supports` | idem |
  - **Custo de virar atributo:** atributo não é alvo de travessia, então nenhuma aresta aponta para ele. Isso só pesa quando a pergunta exige caminhar até a coisa (ex.: "quem usou a chave?"); é o sinal, na análise de erros, para criar um perfil enriquecido.
  - **Alternativas comparadas:**
    - *Ontologia fixa no código (`if type == ...` espalhado):* cada experimento vira refatoração. → Descartada.
    - *Registro em YAML/JSON:* precisa de carregador e validação próprios; os produtores continuam sendo código; perde a checagem de tipos e o autocompletar. Ganho real só se não programadores fossem editar a ontologia. → Descartada; o registro Python já é declarativo.
    - *Tipos criados pelo LLM em tempo de execução (esquema aberto):* *schema drift*; não dá para ablar nem testar, e o `w_Tq` fica indefinido. → Descartada.
    - *Rótulos do Neo4j como ontologia:* infraestrutura extra (§3.2). → Descartada.
    - *Adotar a ontologia completa dos rascunhos já no PoC:* mais extração e mais falhas sem pergunta de benchmark que meça o ganho. → Descartada; os tipos ficam no catálogo, à espera de evidência.

## 10. Milestones

| # | Branch | Entrega |
|---|---|---|
| M0 | `main` | `.gitignore`, `pyproject`/`requirements`, modelos locais configurados, `llm.py` com cache e contadores |
| M1 | `magma-original` | Núcleo fiel (§6): graph, index, temporal, router, fast/slow path, RRF, beam, linearização, C.1/C.2/C.3 + testes unitários |
| M2 | `magma-original` | Sanity no subconjunto do LoCoMo + as 4 ablações da Tab. 4 + métricas da Tab. 3; tag `v-original` |
| M3 | `magma-narrative` | 3 cenários + gabarito + sondas + loader + runner incremental |
| M4 | `magma-narrative` | `narrative/`: claims, `visible_to`, contradição/status, recuperação por viewer, prompts de agente, orquestrador ao vivo |
| M5 | `magma-narrative` | Baselines Full Context + Naive RAG |
| M6 | `magma-narrative` | Métricas, juiz, estatística, tabelas/gráficos |
| M7 | opcional | Benchmarks externos extras (ex.: FactConsolidation, se não entrar na D1) + análise de sensibilidade do embedding + comparação de ontologias (`narrative-min` × perfil enriquecido definido pela análise de erros, D9) |

**Gargalo esperado:** inferência na ingestão (2 chamadas LLM por turno na Branch A; 2–3 na Branch B). As execuções de benchmark rodam em background enquanto o resto é implementado; o cache evita repetir chamadas.

## 11. Referências
- MAGMA (ACL 2026) — https://aclanthology.org/2026.acl-long.1709/ · código: https://github.com/FredJiang0324/MAGMA
- Staying In Character / ReverieMem / KBF-QA — https://arxiv.org/pdf/2606.25632
- Sotopia-ToM — https://arxiv.org/pdf/2605.02307
- MemoryAgentBench (FactConsolidation) — https://arxiv.org/html/2507.05257v1 · https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
- ParliamentBench — https://www.alphaxiv.org/abs/2607.28146
- Mini-Mafia — https://arxiv.org/pdf/2509.23023
- TimeChara — https://arxiv.org/pdf/2405.18027
- CHARM — https://arxiv.org/pdf/2609.01352
- iAgents / InformativeBench — https://web3.arxiv.org/pdf/2406.14928
- FANToM — https://arxiv.org/abs/2310.15421v3 · espelho HF: https://huggingface.co/datasets/TomTraining/TomDatasets
- Diplomacy Deception — https://aclanthology.org/2020.acl-main.353.pdf · HF: https://huggingface.co/datasets/community-datasets/diplomacy_detection
- ExploreToM — https://arxiv.org/abs/2412.12175 · HF: https://huggingface.co/datasets/facebook/ExploreToM
- OpenToM — https://arxiv.org/abs/2402.06044v3
- Hi-ToM — https://huggingface.co/umwyf/Hi-ToM_Dataset
- Dense X Retrieval (Chen et al., 2024) — granularidade de recuperação (proposições × passagens)

---

## 12. Fora do escopo do PoC (melhorias futuras)

Fase atual: prova de conceito. Critério para entrar no PoC: ser exigido pelos requisitos (§2) ou pela fidelidade ao artigo (§1). O que não atende a nenhum dos dois fica aqui, com a análise já feita, para o MVP ou a implementação final.

| Item | Por que fica fora do PoC | Análise |
|---|---|---|
| Segmentação de mensagens longas em vários eventos (diluição) | O artigo não trata; é otimização de recuperação, não requisito. O PoC usa 1 turno = 1 evento | D8 |
| Índice aproximado (HNSW/IVF) ou banco vetorial dedicado | Só compensa a partir de ~10⁵ vetores; o PoC é bem menor | D7 |
| Router **alternativo** por LLM / otimização de prompts (ex.: GEPA) | O router do artigo (intenção WHY/WHEN/ENTITY, parser temporal, keywords, pesos adaptativos) **está no PoC** (§6.3); aqui fica só a variante por LLM, por custo e não determinismo | D2 |
| Hierarquia de nós (agrupamentos episódicos) | O artigo só a menciona (§3.2), sem algoritmo; o `EpisodeNode` do código oficial vem do Nemori, fora do artigo | §1.8 |
| Grafo de crenças explícito (`BELIEVES {confidence}`) | O status relativo ao observador cobre os RFs | §7.4 |
| Desafios narrativos: sarcasmo, boatos, promessas, ToM de 2ª ordem, decaimento de memória, hipotéticos, vínculos sociais, autoridade, `REPLIES_TO`/`ADDRESSED_TO` | Cada um é um projeto à parte | §3.2 |
| Sessões ao vivo como instrumento de **avaliação** (o orquestrador em si está no PoC) | Variância e custo; o PoC mede por replay | §7.5 |
| Baselines do artigo (A-MEM, Nemori, MemoryOS) e Mem0 rodando com o LLM local | Cada um exige integração própria; números publicados usam outro LLM e não são comparáveis | D4 |
| Comparação ampla de embeddings | O PoC fixa um modelo e faz só a checagem com o MiniLM | D8 |
| Banco de grafos dedicado (Neo4j/Memgraph) | Infra extra sem efeito nos RFs; o artigo admite backend em memória. **A persistência está no PoC** (NetworkX + JSON, como o código oficial) | §3.2 |
