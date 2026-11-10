# Design — MAGMA Original + MAGMA Narrativa

> Design doc vivo do projeto: escopo, requisitos, arquitetura, decisões com alternativas consideradas, protocolo de avaliação e plano. Qualquer agente deve conseguir continuar o trabalho a partir daqui.
> As specs concretas de cada módulo (interfaces, contratos e trechos de código) derivam deste documento e ficam ao lado dele, em `docs/`.
>
> **Status:** análise concluída. **Decisão principal:** reimplementar o MAGMA do zero, fiel ao artigo (§5).
> **Decididas (todas):** D1 (benchmarks por requisito; modos `sample` e `report`), D2 (router determinístico com modelo de decisão local), D3 (GLM 5.3 Flash + juiz DeepSeek V4.1 Flash; segundo juiz a escolher), D4 (baselines), D5 (inglês), D6 (sem framework de LLM), D7 (FAISS exato), D8 (Qwen3-Embedding-0.6B), D9 (perfil `narrative`). Premissa da avaliação: comparação controlada entre as implementações do projeto; o artigo é referência (D1).
> **A verificar na implementação:** se o canal temporal do RRF ganha com as `dates_mentioned` além do τ (decidido τ ou datas, §6.4; na menor conversa do LoCoMo só 4 perguntas têm janela e evidência, poucas para concluir; a medição se repete com as demais conversas); se o LongMemEval-KU entra no RF3 (D1).
> **Próximo passo:** commits 1–13 do plano (§10.2) implementados e a menor conversa do LoCoMo construída pelo runner; segue o commit 14 (fachada com a resposta C.2 e o primeiro run `sample`). O plano foi reagrupado de 42 para 26 commits, com o runner antes da recuperação. Em aberto até o portão de custo do commit 15: a forma de conter o custo da ingestão, o provedor do LLM principal (D3) e a construção em paralelo.
> **Regra da §9:** toda decisão registra recomendação, justificativa, alternativas comparadas e por que cada uma foi descartada (material direto para o artigo). Sem tom de roteiro de apresentação.
> **Regra de escopo (PoC):** implementar tudo o que o artigo descreve e tudo o que a extensão exige; só otimizações e extras vão para a §12.
>
> Mapa: §0 escopo · §1 artigo · §2 requisitos · §3 ideias avaliadas · §4 auditoria do código oficial · §5 decisão ·
> §6 Arquitetura A · §7 Arquitetura B · §8 benchmark/métricas · §9 decisões · §10 milestones · §11 referências ·
> §12 fora do escopo do PoC

---

## 0. Escopo e princípios

**Entregas**
1. **Arquitetura A (`magma/`)** — reimplementação fiel do MAGMA (Jiang et al., ACL 2026). A tag `v-original` marca a versão validada.
2. **Arquitetura B (`narrative/`)** — extensão do MAGMA para narrativas multiagente (fato × declaração, visibilidade, contradição), no mesmo repositório, importando e estendendo `magma/`.
3. **Benchmark comparativo** entre as duas arquiteturas e baselines tradicionais de memória.

**Princípios**
- **Fase atual: prova de conceito (PoC).** Roteiro: PoC → MVP → implementação final. O PoC implementa **todos os componentes descritos no artigo** (router de intenção, RRF, beam search, linearização com budgeting, dual-stream, 4 grafos, persistência) e **todos os da extensão** (RF1–RF3 + agentes e orquestrador). Ficam para depois só otimizações e extras que nem o artigo nem os requisitos pedem (§12).
- **Baixa complexidade:** implementar só o necessário para os requisitos (§2); o resto vira trabalhos futuros.
- **Reprodutibilidade:** `temperature=0`, `seed` fixo, cache de chamadas LLM em disco, hiperparâmetros num único arquivo.
- **Modelos remotos atuais** via endpoint OpenAI-compatível (D3); o mesmo cliente aceita endpoint local.
- **Console interativo em cada arquitetura** (§6.2, §7.5): conversar com a memória e com os personagens, com o rastreamento da recuperação visível. Serve de demonstração e de teste de integração; as métricas continuam vindo do runner por replay (§8).
- **Avaliação:** comparação controlada entre as implementações do projeto, todas com o mesmo LLM; do artigo comparam-se tendências, não números absolutos (D1).
- **Evidência principal vem de benchmarks públicos e estabelecidos** (validados e citados na literatura), não de dados criados pelo próprio projeto. Conjunto próprio só como diagnóstico complementar (D1).
- **LoCoMo/LongMemEval** não têm narrativa multiagente, segredos nem mentiras; servem para fidelidade da Arquitetura A e para a retenção (H2), não para os RFs narrativos.
- **Roteamento de intenção:** determinístico, como no artigo e no código oficial (D2).

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
- **Fast path (Alg. 2):** segmenta evento → aresta TEMPORAL com o último nó → embedding → indexa no VDB → enfileira id. **Sem LLM.**
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
- **A Tab. 6 lista parâmetros que o algoritmo não usa:** `w_keyword (Fusion)` (a Eq. 4 não tem peso), `Sim. Threshold` e `Drop Threshold` (o Alg. 1 só corta vizinhos pelo `BeamWidth` e pelo `Budget`). O único limiar de similaridade definido no texto é o `θ_sim` das arestas `SEMANTIC` (§3.2), sem valor.
- **Canais do RRF:** a Eq. 4 soma três canais (`vec`, `key`, `time`), mas a linha 2 do Alg. 1 só passa dois (`V.SEARCH(q)`, `K.SEARCH(q_key)`).
- **"w/o Adaptive Policy" (Tab. 4)** não é definida: o texto só diz que a busca "degenera numa caminhada estática pelo grafo".
- **Limiares absolutos de cosseno dependem do modelo de embedding:** os valores da Tab. 6 foram calibrados para o MiniLM e não se transferem para outro modelo.
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
| 5 tipos de nó primitivos | **REVISTO** (D9) | Perfil `narrative` (D9): `Event` (fato observado; ação física em `action_text`), `Claim` (declaração), `Entity` (inclui agentes, `kind=agent`). O destino de cada primitivo está na D9 |
| ~20 tipos de aresta | **REVISTO** (D9) | 9 tipos no `narrative` (§7.2); as relações de papel viram `MENTIONS{role}` e as demais têm destino registrado na D9 |
| `REPLIES_TO`, `ADDRESSED_TO` | **CORTAR** | Não exigidos pelos RFs |
| `SUPERSEDES` | **MANTER (núcleo)** | Distingue atualização legítima ou mudança de versão do autor de contradição entre autores (D9); exigida pelos benchmarks de atualização do RF3 |
| Sarcasmo, boatos, promessas, ToM de 2ª ordem, decaimento, hipotéticos, vínculos sociais, autoridade | **CORTAR** → trabalhos futuros | Cada um é um projeto à parte |
| Lorebook / Seed Graph | **MÍNIMO** | Só um JSON de cenário (personagens, segredos, fatos iniciais com `visible_to`) |
| Neo4j / Memgraph / Cypher | **CORTAR** | Infra extra; o artigo (§3.5) admite backend em memória. NetworkX + índice vetorial em memória: zero infra, reprodutível, mesmo backend nas duas arquitetches (variável controlada) |
| FastAPI / .NET | **CORTAR** | Não precisa de servidor: consoles de terminal (§6.2, §7.5) + scripts de benchmark |
| LangChain / LangGraph / PydanticAI no núcleo | **CORTAR** | Abstraem justamente o que é a contribuição (roteamento, recuperação, consolidação, prompts) e atrapalham a medição de tokens/latência (D6) |
| LlamaIndex (ou LangChain) | **SÓ NO BASELINE C2** | É o "RAG de terceiros" exigido; isolado em `baselines/naive_rag.py` |
| LLMs proprietários antigos (GPT-4o, Claude 3.5, Gemini 1.5) | **SUBSTITUIR** | LLM atual via endpoint OpenAI-compatível (D3) |

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
| LoCoMo (`data/locomo10.json`) + `load_dataset.py` | Opcional, só para o *sanity check* da Arquitetura A |
| Prompt do juiz C.3 (do artigo, não o código) | Reescrever **sem** atalho de categoria |
| `memory/temporal_parser.py` (regex + timedelta, específico do LoCoMo) | Substituir por `dateparser` (determinístico, justificável) |
| Resto (answer_formatter, best_of_n, nós episode/session/narrative, script do LongMemEval, A-MEM em `utils/memory_layer.py`) | **Descartar** |

### 4.4 Pontos de entrada
- **`main.py` não executa desde o commit inicial.** A classe `TRGSystem` (modos `build`, `query` e `test`) foi escrita para uma API que o resto do código não tem:
  - os construtores recebem argumentos que não existem: `MemoryBuilder(trg_memory=…, model=…)` (`main.py:58`; o real recebe `cache_dir, llm_model`, `memory/memory_builder.py:33`), `QueryEngine(trg=…, model=…)` (`main.py:64`; o real recebe `trg_memory, node_index`, `memory/query_engine.py:30`), `TestHarness(trg=…, …)` (`main.py:71`; `memory/test_harness.py:32`) e `Evaluator(model=…)` (`main.py:77`; `memory/evaluator.py:31`);
  - chama métodos que não existem: `format_context`, `run_tests`, `evaluate_results`, `save_to_file` e `load_from_file` (os reais são `save`/`load`, `memory/trg_memory.py:916` e `:934`).
- **Nenhum arquivo importa `TRGSystem`**, e o README só documenta os scripts de benchmark.
- **As entradas que funcionam são os scripts de benchmark** (`test_fixed_memory.py:335` e `test_longmemeval_chunked.py`), que montam os componentes à mão (`test_fixed_memory.py:448-498`).
- **Não há interface interativa:** nenhum ponto do código lê entrada do usuário.
- → Este projeto expõe a memória por um console em cada arquitetura (§6.2, §7.5), além do runner.

---

## 5. Decisão: reaproveitar vs reescrever → **REESCREVER DO ZERO** (fiel ao artigo)

Implementar o MAGMA *a partir do artigo* (Eqs. 3–7, Algoritmos 1–3, Tabela 6, Prompts C.1–C.3) num pacote novo e pequeno, sem importar código do repositório oficial.

**Justificativas**
1. **Fidelidade:** os componentes centrais do artigo (beam search, router de 3 intenções, dual-stream, arestas de entidade e causais por LLM) não estão ativos no código oficial (§4.1). Estendê-lo significaria estender outra arquitetura, não o MAGMA descrito.
2. **Validade experimental:** o código oficial usa rótulos oracle e mapas fixos do LoCoMo (§4.1 #6–#8). Uma comparação Original × Narrativa feita sobre ele ficaria contaminada.
3. **Variável controlada:** as duas arquiteturas compartilham exatamente o mesmo núcleo; a única variável independente é a extensão narrativa.
4. **Custo:** o núcleo fiel cabe em ~800–1200 linhas, contra ~14k linhas de heurísticas para entender e consertar.
5. **Licença:** o MIT permite reaproveitar ideias e estrutura; o repositório oficial é citado como referência, com as divergências documentadas (§4).

**Risco e mitigação:** o artigo é omisso em vários pontos (§1.8). Cada lacuna recebe valor escolhido + justificativa (§6.4).

---

## 6. Arquitetura A (`magma/`): reimplementação fiel

### 6.1 Princípios
- Cada módulo mapeia para uma equação/algoritmo e cita a seção no docstring (ex.: `# Eq. 5, §3.3`).
- **Sem rótulos oracle.** Nenhum código recebe categoria ou gabarito da pergunta.
- **Determinismo:** `temperature=0`, `seed` fixo, cache em disco das chamadas LLM (chave = hash do prompt + modelo).
- **Backend em memória** (NetworkX + NumPy), permitido pelo §3.5 do artigo.
- **Janela de contexto explícita:** `llm.py` conhece a janela do modelo configurado, acusa quando um prompt passa dela (alguns servidores descartam o início do texto sem erro, ex.: Ollama com o `num_ctx` padrão) e registra os tokens de entrada, saída e raciocínio devolvidos pela API. Vale para todas as chamadas e todas as arquiteturas, com atenção especial ao baseline Full Context.
- Todos os hiperparâmetros num único `config.py` (espelho da Tabela 6) → ablações por flag.

### 6.2 Layout do pacote
```
magma/
  config.py      # Tabela 6 + valores escolhidos onde o artigo é omisso (§6.4)
  llm.py         # cliente OpenAI-compatível (LLM_API_KEY/LLM_MODEL/LLM_BASE_URL), cache em disco,
                 # contabiliza prompt/completion tokens e latência por chamada; grava prompt e resposta crus
                 # de toda chamada em runs/<run_id>/llm_calls.jsonl (D6)
  embed.py       # sentence-transformers (em processo, sem HTTP) carregando o modelo da D8: Qwen3-Embedding-0.6B
                 # (prompt de instrução só na query, vetores normalizados).
                 # O limite de corte é o `max_seq_length` do MODELO (Qwen3: janela de 32k; modelos curtos como o MiniLM: 256 word-pieces),
                 # não da biblioteca: loga o valor ao carregar e avisa quando um texto passa dele (o corte é silencioso)
  schema.py      # registro declarativo de tipos de nó e aresta + perfis (D9); perfil `magma` = nós EVENT, ENTITY; arestas NEXT (TEMPORAL), SIMILAR_TO (SEMANTIC), MENTIONS (ENTITY), CAUSES (CAUSAL)
  graph.py       # núcleo agnóstico a tipo: Node{id,type,text,τ,attrs,visible_to}, Edge{source,target,type,weight,attrs},
                 # interface GraphStore, tipos validados contra o perfil ativo do schema.py; persistência em JSON
                 # (save_graph/load_graph sobre qualquer GraphStore), como o código oficial (`graph_db.py:666-706`).
                 # O vetor v_i fica no índice vetorial (index.py), não no nó
  networkx_graph.py  # GraphStore sobre NetworkX MultiDiGraph (aresta com chave = tipo); único módulo que importa networkx
  index.py       # bases VectorIndex e LexicalIndex (busca com conjunto de ids permitidos, aplicado antes do ranking),
                 # tokenizador comum a textos e consultas, persistência dos vetores (.npz)
  faiss_index.py # VectorIndex sobre FAISS IndexFlatIP (D7); único módulo que importa faiss
  bm25_index.py  # LexicalIndex sobre rank_bm25 (Okapi); único módulo que importa rank_bm25
  temporal.py    # tagger temporal (dateparser): expressão → janela do tamanho dela (dia, semana, mês, ano);
                 # na pergunta, janela [τs, τe]; no turno, resolve dates_mentioned com base no timestamp do evento
  router.py      # Stage 1: intenção T_q∈{WHY,WHEN,ENTITY} + janela [τs,τe] + v_q + q_key
  intent_model.py # modelo de decisão que rotula a intenção (GLiNER2); único módulo que importa gliner2
  ingest.py      # Alg.2 Fast Path (sem LLM) + Alg.3 Slow Path (fila + worker; LLM) + prompt C.1
  retrieve.py    # Stage 2 RRF (Eq.4), Stage 3 beam search (Alg.1, Eq.5-6), Stage 4 linearização (Eq.7) + budgeting
  qa.py          # prompt C.2 com instrução dinâmica gerada pelo router
  memory.py      # ABC MemorySystem (add_turn / consolidate / query → resposta + rastreamento + uso do LLM + latência; usage; save)
                 # e a fachada MagmaMemory que a implementa (fast path + slow path; a resposta entra com a recuperação e o C.2)
  cli.py         # console (python -m magma): chat, ingest, ask; /why imprime o rastreamento da última consulta
bench/           # harness, métricas, juiz C.3, estatística (compartilhado com a Arquitetura B)
  benchmark.py                     # formato único do runner: Conversation (turnos datados, momento da pergunta) e Question (categoria, gabarito)
  locomo.py                        # leitor e adaptador do LoCoMo (sessões em ordem, turnos com legenda da imagem, 5 categorias,
                                  # gabarito "Unanswerable" nas adversariais)
  runner.py                        # runner de benchmark contra MemorySystem: amostra estratificada com seed, modos sample/report,
                                  # piloto de custo e teto, --build-only, juiz, F1/BLEU-1, Tab. 3; grava em results/<run_id>/
  judge.py                         # juiz com o prompt C.3 (modelo configurável, JUDGE_*)
  lexical_metrics.py               # F1 por token e BLEU-1 (§4.1 do artigo)
  calibrate_semantic_threshold.py  # mede a distribuição de cossenos entre turnos e o efeito de cada θ_sim no fast path;
                                  # rodar de novo ao trocar o embedding (python -m bench.calibrate_semantic_threshold --locomo ...)
  measure_temporal_windows.py      # cobertura do tagger temporal nas perguntas do LoCoMo (janelas, filtro, granularidade)
  probe_reasoning.py               # tokens de raciocínio por forma de pedir (esforço, orçamento) e por provedor
  measure_consolidation.py         # slow path numa conversa do LoCoMo com LLM e embedding reais: entidades, datas,
                                  # tamanho da vizinhança causal, posição das causas escolhidas, falhas e tokens
  measure_router.py                # classificadores de intenção do Stage 1 nas 1.986 perguntas do LoCoMo, por categoria
  measure_retrieval.py             # Stages 1–4 sobre uma memória construída, sem LLM: evidência no canal temporal (τ × τ ou datas),
                                  # nas âncoras, nos visitados e no contexto; disparos do drop threshold; cossenos; tamanho do contexto
baselines/       # full_context.py, naive_rag.py
tests/           # testes unitários de RRF, beam search e linearização com grafo de brinquedo
```

### 6.3 Fluxos
- **`add_turn(speaker, text, timestamp)`** — Fast Path (Alg. 2): cria `EventNode` (1 turno = 1 evento), aresta `TEMPORAL` com o último nó, embedding + BM25, arestas `SEMANTIC` (cos > θ_sim, top-5 vizinhos), enfileira o id. **Nenhuma chamada LLM.**
- **`consolidate()`** — Slow Path (Alg. 3): para cada id da fila, em ordem → prompt C.1 (entidades, tópico, fatos, datas, resumo; o *Context* é o resumo do evento anterior) → cria/une `EntityNode` + arestas `ENTITY`; resolve `dates_mentioned` via `temporal.py` com o τ do evento como referência e grava em `A_i`; eventos **anteriores** da vizinhança 2-hop → prompt causal → arestas `CAUSAL` (causa → efeito) com confiança > δ.
  - Roda como **thread de fundo real** (fiel ao artigo), mas o benchmark chama `consolidate(drain=True)` antes de cada sonda → resultado determinístico.
- **Fronteira `MemorySystem`:** interface única de sistema de memória (`add_turn`, `consolidate`, `query` → resposta + rastreamento). É uma ABC com as regras que toda implementação cumpre: turnos em ordem de tempo, e a `query` da base mede a latência e o uso do LLM de cada consulta (delta do `usage()`), o que nenhuma implementação consegue pular; `save` grava o conteúdo da memória para inspeção e para o teste de regressão. O runner, as métricas e os consoles dependem só dela, nunca do interior de `magma/`. A fachada `MagmaMemory`, a Arquitetura B e os baselines C1 e C2 a implementam, o que os torna intercambiáveis no benchmark e permite separar o código das arquiteturas no futuro sem refatorar o benchmark (§12).
- **`query(q)`** — Stage 1→4 (Alg. 1): router → RRF(vec, key, time) top-K âncoras → beam search com `score_v = score_u·γ + exp(λ1·w_Tqᵀ·1_r + λ2·sim)` → ordenação (WHEN: τ; WHY: sort topológico em `CAUSAL`; ENTITY: score) → `<t:τ> conteúdo <ref:id>` → nós de baixa saliência viram `"...N intermediate events..."` até caber no orçamento de tokens → prompt C.2.
  - Além do contexto, `query` devolve um **rastreamento**: intenção e janela do router, âncoras com a posição em cada canal do RRF, caminhos do beam com tipo de aresta e pontuação, nós podados pelo budgeting e tokens. O console imprime esse rastreamento e o runner o grava junto de cada resposta, o que permite estudos de caso no estilo do Ap. E do artigo.
- **Console (`cli.py`)** — mesma fachada, sem lógica própria de memória:
  - `chat`: cada fala do usuário passa por `add_turn`; a resposta vem de `query` + C.2 e também é ingerida como turno do assistente. A consolidação roda em segundo plano, como no artigo.
  - `ingest <arquivo>` (transcrição em turnos `speaker`, `text`, `time`) e `ask "<pergunta>"` sobre um grafo salvo.
  - Comandos: `/why` (rastreamento da última consulta), `/stats` (nós e arestas por camada, fila do slow path, tokens), `/date` (data simulada da sessão, para perguntas temporais), `/save` e `/load`.
  - `--script <arquivo>` lê os comandos de um arquivo em vez do teclado. Com o cache do `llm.py`, a mesma sessão se repete sem custo; é a base do teste de integração e de demonstrações reproduzíveis.

### 6.4 Decisões onde o artigo é omisso
| Lacuna | Escolha | Justificativa |
|---|---|---|
| O que é um "evento" | 1 turno = 1 `EventNode` | Mais simples; igual ao uso em LoCoMo |
| C.1 usa LLM, mas o fast path não pode | C.1 roda no **slow path** | Respeita "no blocking LLM" do fast path |
| Classificador "leve" de intenção | Modelo de decisão local (`fastino/GLiNER2.5-Decide`, encoder de uma passada) sobre a pergunta sem as expressões de data, com uma descrição por intenção | Determinístico, local, sem custo por pergunta; separa o que a pergunta pede do contexto temporal, que já vai para a janela (D2) |
| Instrução dinâmica C.2 (4 categorias) × router (3 intenções) | WHEN→Temporal, WHY→Multi-hop/Inferência, ENTITY→Single-hop; "both/all/compare" → Multi-hop | Sem oracle; uma única fonte de roteamento |
| Canal lexical ("keywords esparsas", sem método declarado) | **BM25 Okapi** (`rank_bm25`), `k1 = 1.5`, `b = 0.75`, piso de IDF `ε = 0.25`; tokens = palavras em minúsculas (`\w+`) sem stopwords do inglês, sem stemming; mesmo tokenizador para textos e consultas | Padrão consolidado de busca lexical, com os valores usuais; o código oficial usa um índice invertido próprio de unigramas e bigramas, sem fundamentação publicada |
| Parser temporal ("temporal tagger" sem método nem granularidade) | `dateparser` com a data de referência (pergunta: fim da conversa; turno: timestamp do evento). Cada expressão vira a **janela do seu tamanho** ("yesterday" = o dia, "last week" = a semana de segunda a domingo, "in June" = o mês, "2022" = o ano); data ambígua lida como passada na pergunta e no período corrente no turno; na pergunta, só valem trechos com palavra temporal ou ano de 4 dígitos, e a janela cobre as menções mais específicas (o ano de "August 11 and August 15 2023" não alarga a janela) | Determinístico e genérico; o código oficial usa regex e `timedelta` específicos do LoCoMo. Expressões compostas ("two weeks before August 11") e vagas ("a few days ago") ficam fora: limitação registrada |
| `w_phrase` | Peso da aresta `SEMANTIC` (similaridade entre trechos) | Único tipo de aresta sem peso declarado |
| `w_keyword (Fusion)` 2.0–5.0 | **RRF ponderado**: `score(n) = Σ_c w_c / (k + r_c(n))`, com `w_keyword = 2.0` e `w_vec = w_time = 1.0` | O nome diz "fusão"; o canal lexical cobre nomes próprios e termos raros, em que o embedding falha. Ponta baixa do intervalo para não dominar os outros canais. Peso dentro do canal lexical (como no código oficial) não muda a fusão, que só usa a posição. Teste de sensibilidade opcional: RRF sem peso (Eq. 4 literal) |
| Matriz `w_Tq` | Pontos dentro dos intervalos da Tab. 6 (ex.: WHY: causal 5.0, temp 1.5, ent 2.5, sem 2.5; WHEN: temp 4.0, causal 3.0, ent 2.5, sem 2.5; ENTITY: ent 6.0, causal 3.0, sem 2.5, temp 0.5) | Respeita os intervalos publicados |
| λ2 | 0.5 (meio do intervalo 0.3–0.7) | — |
| γ (decay) | 0.9 | Não declarado |
| BeamWidth | 10 | Valor do código-referência |
| δ (limiar causal) | 0.5 sobre a confiança emitida pelo LLM; a aresta exige confiança **acima** de δ, como em `S(n_j \| n_i, q) > δ` (§3.2) | Não declarado |
| Resolução de entidades (o slow path "cria ou une" entidades, sem regra declarada) | Mesma entidade quando o nome coincide em minúsculas, com espaços colapsados e sem o possessivo ("Caroline's" = "caroline"); o nó guarda a primeira grafia vista | Determinístico e sem custo. Apelidos ("Mel" × "Melanie") não são unidos (limitação). Descartadas: texto exato (separa "Caroline" de "caroline") e alias por LLM ou embedding (uma chamada ou um limiar novo por entidade, sem valor no artigo) |
| Candidatas do prompt causal (`GETNEIGHBORHOOD(n_t, hops=2)`, sem tamanho declarado) | Os eventos **anteriores** a `n_t` na vizinhança de 2 saltos (todas as camadas), todos eles; a busca não passa por eventos posteriores. Ordem no prompt: 1 salto antes de 2, depois do mais recente ao mais antigo. Limite opcional no `config.py` (`causal_candidate_limit`, padrão sem limite) | Literal ao Alg. 3. Só eventos anteriores porque a causa vem antes do efeito, o que mantém o grafo causal acíclico (ordenação topológica do Stage 4); sem passar por eventos posteriores, a vizinhança não depende de quanto o fast path adiantou. Por uma entidade frequente, o segundo salto alcança boa parte da conversa: o prompt cresce com ela. Descartado no perfil `magma`: limite fixo de candidatas (cortaria causas distantes ligadas pela entidade) |
| Prompt causal (o Alg. 3 só diz que o LLM infere as arestas) | Prompt próprio no estilo do C.1: lista os eventos anteriores que causaram, levaram a ou motivaram o novo, com confiança de 0 a 1 e um motivo curto; ids fora da lista são ignorados | O artigo não traz o prompt; motivação conta como causa (D9) |
| Saída inválida do LLM no slow path | Sem nova tentativa: o evento recebe `consolidation_errors` e a consolidação segue; a taxa entra na medição | Repetir a requisição devolveria a resposta do cache; repetir com o erro na conversa gastaria chamadas fora da medição |
| θ_sim (aresta SEMANTIC) = `Sim. Threshold` da Tab. 6 | Limiar da aresta `SEMANTIC`, com no máximo 5 vizinhos por nó. **θ_sim = 0.75**, medido com o Qwen3 nas duas primeiras conversas do LoCoMo (vetor de "Falante: texto", como o `ENCODER(n.c)` do Alg. 2): entre turnos de uma conversa, mediana 0.54, p95 0.71, p99 0.78. Com 0.75 (≈ p97–98), cada evento novo ganha em média ~2,2–2,6 arestas e ~70% ganham ao menos uma; com 0.6, 92% batiam o limite de 5 e o limiar não filtrava. Nenhum valor separa assunto de estilo nesse dataset (pares de "Thanks, Caroline!" passam de 0.8); a relevância à pergunta fica com o `λ2·sim` da travessia. Regra para outro embedding: ≈ percentil 97–98 dos cossenos entre turnos de uma conversa, medido com `bench/calibrate_semantic_threshold.py` | Único limiar de similaridade definido no texto (§3.2). O intervalo 0.10–0.30 é do MiniLM e, com outro embedding, ligaria quase todos os pares. Descartadas: piso de similaridade na travessia (como no código oficial; duplica o termo `λ2·sim` da Eq. 5 e não está no Alg. 1) e piso do canal vetorial (sem apoio no texto) |
| Drop threshold 0.15 | Piso absoluto: descarta o vizinho com `sim(n_j,q) < 0.15`, para eventos e entidades. Conferido na menor conversa do LoCoMo (105 perguntas, `bench/measure_retrieval.py`): o cosseno pergunta–evento tem p1 = 0,17 e mediana 0,43, e pergunta–nome de entidade p1 = 0,14 e mediana 0,28; 0,4% dos pares pergunta–evento e 1,6% dos pares pergunta–entidade ficam abaixo de 0,15, e o piso descarta 0,29% dos passos avaliados na travessia | Leitura mais simples de "drop"; com embeddings de cosseno alto funciona como rede de segurança, e a medição confirma esse papel: só corta vizinhos quase sem relação com a pergunta. Descartadas: queda relativa por passo (como no código oficial; regra que o artigo não descreve) e não usar (Alg. 1 literal; quase equivalente, já que o piso raramente dispara) |
| Instrução do embedding (Qwen3) | Só na query: `Instruct: Given a question about a conversation, retrieve the memory entries that answer it
Query:`; textos armazenados sem prefixo | Formato do modelo; a tarefa descreve o uso real (pergunta → entrada de memória) |
| Orçamento de contexto | ~2.5k tokens, contados com o tokenizador do modelo de embedding (Qwen3) | Na ordem dos 3.37k tokens/query da Tab. 3. O tokenizador já está carregado, é local e determinístico; não é o do LLM, então o orçamento é aproximado, e os tokens reais de cada consulta vêm da API (Tab. 3). Descartados: `tiktoken` (o do gpt-4o-mini do artigo; uma dependência a mais para uma aproximação equivalente) e caracteres ÷ 4 (impreciso) |
| Canal temporal do RRF (a Eq. 4 soma `r_time`, sem defini-lo; o Stage 1 e a Fig. 2 chamam o sinal de filtro) | **Canal, não filtro duro.** Entra no canal o evento com τ dentro da janela `[τs, τe]` da pergunta **ou** com alguma `dates_mentioned` resolvida que se sobrepõe a ela; todos empatam no **posto médio** `r_time = (N + 1)/2` (N = eventos na janela), e quem está fora não pontua. Pergunta sem janela: canal vazio. Chave `time_channel_uses_mentioned_dates` para usar só o τ | τ é o tempo da fala, as datas citadas são o tempo do fato ("we went camping last year", dito em 2023, para uma pergunta sobre 2022). Um filtro diz dentro ou fora, sem ordem; o posto médio faz uma janela larga, que informa pouco, pesar pouco (40 eventos: 1/80,5 cada; 500 eventos: 1/310,5), sem parâmetro novo. Como canal, um erro do parser só tira um bônus. Descartados: só o τ (perde o tempo do fato), só as datas (perde o tempo da fala), ordenar a janela por similaridade (contaria o vetorial duas vezes), todos no posto 1 (uma janela de um ano valeria o 1º do vetorial) e filtro duro nas âncoras (apagaria a evidência num erro do parser, e o termo `time` da Eq. 4 não mudaria nada) |
| Profundidade dos canais e número de âncoras (Tab. 6: "Vector Top-K 20") | Os canais vetorial e lexical devolvem os 20 melhores; o canal temporal, todos os eventos da janela; o RRF fica com as **20 âncoras** de maior pontuação (empate pela ordem do grafo). Um acerto numa faceta conta para a sua âncora, com o melhor posto entre elas | Um único valor da Tab. 6 para a lista de cada canal e para o top-K da Eq. 4; o top-K conta âncoras únicas (regra das facetas, D9) |
| `sim(n_j, q)` de um nó de entidade (a Eq. 5 o exige em todo vizinho; a Eq. 3 só dá vetor ao evento) | A entidade é pontuada pela **codificação do nome** (espaço dos textos armazenados, sem instrução), calculada na primeira consulta que a alcança e guardada; o piso de similaridade vale como nos eventos. A entidade conta no orçamento de nós da travessia, mas **não é escrita** no contexto: os eventos que a citam levam o conteúdo | É a regra do Alg. 2 (`v ← ENCODER(n.c)`) aplicada ao conteúdo da entidade, o nome; a ingestão não muda. Com sim = 0, todas as entidades empatariam em `exp(λ1·w_entity)` e o piso descartaria todas (a camada morreria). O código oficial não tem nós de entidade: liga direto os eventos que citam a mesma entidade (`SAME_ENTITY`). Descartados: sim = 0 isento do piso (a escolha entre entidades ignoraria a pergunta), salto colapsado evento → evento (foge do grafo do artigo, com a entidade como nó abstrato) e o vetor emprestado dos eventos que a citam (a média de um falante citado em todos os turnos é a média da conversa) |
| Score inicial das âncoras no Alg. 1 e saliência no budgeting ("relevance scores S(n_j\|n_i, q) computed in Eq. (5)") | Score inicial = o **RRF da âncora**. Saliência de um nó alcançado = o `s_uv` da transição que o manteve no beam; no budgeting, os nós alcançados são cortados primeiro, do menor `s_uv` para o maior, e as âncoras só depois, do menor RRF para o maior | O RRF é a única pontuação que a âncora tem. Ele fica entre ~0,01 e ~0,07, e qualquer `s_uv` vale de e^0,5 a e^6: na travessia, a parcela da âncora no `score_v` é desprezível (todas as âncoras são expandidas de qualquer forma), e no budgeting as duas escalas não são comparadas. Descartados: 1/(k+1) constante (como no beam morto do código oficial; descarta a ordem do RRF) e o score cumulativo para todos os nós (cortaria as âncoras, escolhidas pelos três canais, antes dos nós a 5 saltos) |
| Ordenação do Stage 4 por intenção e "brevity codes" | WHEN: por τ. WHY: ordenação topológica (Kahn) das arestas `CAUSAL` entre os eventos recuperados, com o evento pronto mais antigo primeiro (empate pela ordem do grafo); sem a camada causal (ablação), cai na ordem por τ. ENTITY: pela saliência. Acima do orçamento, o evento menos saliente é dobrado, um de cada vez, até caber; eventos dobrados consecutivos viram um único `...N intermediate events...` | O artigo só define WHEN e WHY; ENTITY segue a mesma saliência do budgeting. No LoCoMo os turnos de uma sessão têm o mesmo τ, e o desempate pela ordem do grafo mantém a ordem da conversa |
| Async | Thread real + `drain` antes das sondas | Fidelidade + reprodutibilidade |
| Imagens nos turnos do LoCoMo (o dataset traz a legenda BLIP de cada foto compartilhada) | A legenda entra no texto do evento: `[Image: <legenda>] <texto>` | Igual ao carregador oficial e ao estudo de caso do Ap. E (o "nó de foto" é a legenda); perguntas sobre o conteúdo de fotos dependem dela. Descartado: só o texto falado (tornaria essas perguntas inrespondíveis) |
| Datas no LoCoMo (o dataset data sessões, não turnos) | Cada turno recebe a data de início da sessão; as perguntas são feitas na data da última sessão (referência da janela temporal) | Única data disponível por turno; as perguntas são feitas depois da conversa inteira |
| Perguntas adversariais (cat. 5, sem resposta no dataset) | Gabarito `Unanswerable`, julgado pelo C.3 como as demais perguntas | A regra 3 da rubrica C.3 trata exatamente esse gabarito. Descartado: o atalho do código oficial (1,0 sem chamar o juiz quando a resposta contém "Information not found"), regra lexical fora do C.3 |
| F1 e BLEU-1 (§4.1, sem implementação declarada) | Tokens normalizados (minúsculas, sem pontuação e sem artigos); F1 por token; BLEU-1 = precisão de unigramas com corte × penalidade de brevidade; sem *stemming* nem suavização | Definições usuais em QA; reportados ao lado do juiz, como no artigo |
| Métricas da Tab. 3 (sem definição operacional) | Build: tempo de parede de `add_turn` de todos os turnos + `consolidate`. Tokens/consulta: entrada + saída do LLM da memória durante a `query`. Latência: tempo de parede da `query`, só sobre consultas sem chamada servida pelo cache | O cache devolve em milissegundos; a latência das chamadas do cache (a original) é reportada à parte |
| O que uma ablação desliga (Tabs. 4 e 5; o texto diz "selectively disabling edge types and traversal mechanisms") | As arestas das camadas removidas **não são percorridas** na travessia (não basta zerar o peso: `exp(0 + λ2·sim)` continua positivo). O canal temporal do RRF e a ordenação por τ do Stage 4 continuam | Os nomes das ablações falam das arestas ("Temporal Backbone", "Causal Links", "Entity Links") e a Tab. 5 chama a coluna de "Graph Configuration"; o canal temporal é um índice de busca sobre o τ e as datas, não uma camada do grafo. Descartado: tirar também o canal temporal em "sem temporal" e nas variantes de grafo único (juntaria duas peças numa ablação só) |
| "w/o Adaptive Policy" (Tab. 4) | Mesmo `w_Tq` para toda pergunta: a média das três linhas da matriz, com todas as camadas ativas | Isola só a adaptação à intenção, mantendo a importância média de cada camada. Descartadas: pesos de uma única intenção (arbitrário) e `λ1 = 0` ou travessia sem pontuação (mudam mais que a política). Segunda ablação opcional: todos os pesos iguais |

### 6.5 Validação da Arquitetura A (não é o benchmark principal)
- Testes unitários com grafo de brinquedo (RRF, beam, ordenação topológica).
- **LoCoMo** nos dois modos do runner (D1); no `report`, as 1.986 perguntas nas 5 categorias. Reproduz a Tab. 1 (MAGMA × Full Context), as métricas da Tab. 3 (tempo de build, tokens/query, latência), as **quatro ablações da Tab. 4** (sem adaptive policy, sem causal, sem temporal, sem entity) e as **três de grafo único da Tab. 5** (só causal, só temporal, só entidade), todas como flags do `config.py`, aplicadas na consulta sobre o mesmo grafo. Sem o router adaptativo não há como validar a principal alegação de latência e acurácia do artigo.
- **Testes de sensibilidade opcionais** (na consulta, sobre o mesmo grafo): RRF sem peso no canal lexical e "w/o Adaptive Policy" com todos os pesos iguais (§6.4).
- **Objetivo:** verificar se os achados qualitativos do artigo se mantêm com um LLM atual (D1). **Não** é objetivo reproduzir os números absolutos (0.700).

---

## 7. Arquitetura B (`narrative/`): extensão mínima para RF1–RF3

Arquitetura B = pacote `narrative/` sobre o `magma/`, no mesmo repositório. O perfil `magma` continua com o comportamento da tag `v-original`, garantido por teste de regressão (§10.1). O runner do benchmark na Arquitetura B executa todas as arquiteturas (original, narrativa, baselines) com o mesmo LLM, embedding e prompts de agente → a única variável independente é o módulo de memória.

### 7.1 Ideia central (RF1): o *ato de fala* é fato; o *conteúdo* é alegação
- "B disse X para A e C no turno 12" é um **fato** (Event observável por quem estava presente).
- "X" é uma **alegação** (`Claim`) com status epistêmico próprio, que pode ser falsa.
- Isso resolve o *graph poisoning* já com 3 tipos de nó: mentiras ficam no grafo, mas nunca viram fato. Mais primitivos são possíveis sem refatorar (perfis da D9).

### 7.2 Esquema (perfil `narrative`, D9)
Os tipos abaixo são declarados em `narrative/schema.py`, que estende o perfil `magma` da Arquitetura A. O perfil `narrative-min` é o mesmo esquema sem `role` e sem `Claim.type` (ablação, §7.6). O `Event` funciona como **âncora** do turno; a `Claim` é uma **faceta** pendurada nele pela aresta de ancoragem `STATED` (D9).

**Nós**
| Tipo | Campos novos | Origem |
|---|---|---|
| `EventNode` (do artigo) | `actor`, `visible_to: set[agent] \| {"*"}`, `turn`, `action_text` (parte observável: gesto, narração), `speech_text` (fala); o conteúdo `c_i` continua sendo o turno inteiro | turno (separação determinística no fast path, D9) |
| `ClaimNode` (novo) | `statement`, `speaker`, `audience` (= `visible_to`), `about: [entidades]`, `type ∈ {assertion, denial}`, `status ∈ {UNVERIFIED, SUPPORTED, CONTESTED, REFUTED, RETRACTED}` (calculado por viewer) | extração |
| `EntityNode` (do artigo) | `kind` em texto livre (ex.: `agent`, `animal`, `feitiço`, `implante`); o único valor lido por mecanismo é `agent` | extração (o ator do turno vem do fast path) |

**Arestas** (8 nas 4 camadas do artigo + 1 de ancoragem; 9 tipos)
| Camada | Aresta | Semântica |
|---|---|---|
| Ancoragem (`anchor=True`) | `STATED` | evento de fala → claim que ele enuncia. Sem camada nem peso: o beam expande âncora e facetas juntas, sem custo de salto. O autor fica no atributo `speaker` |
| Temporal | `NEXT` (artigo) | cronologia |
| Temporal | `SUPERSEDES` | versão nova → versão anterior: claim do mesmo autor (anterior fica `RETRACTED`) ou fato atualizado (anterior ganha `superseded_by`). Obrigatória (D9) |
| Semântica | `SIMILAR_TO` (artigo) | cos > θ_sim |
| Entidade | `MENTIONS` (artigo) | evento/claim → entidade, com `role ∈ {actor, target, instrument, location, recipient, other}` (D9); a menção evento → ator (`role=actor`) é criada pelo fast path |
| Causal | `CAUSES` (artigo) | inferida no slow path; inclui motivação (causa mental) |
| Causal | `INVALIDATES` | **fato** (`action_text` do evento) → claim que ele refuta |
| Causal | `SUPPORTS` | fato → claim que ele confirma |
| Causal | `CONTRADICTS` | claim ↔ claim incompatíveis de autores diferentes |

### 7.3 Escrita
- **Fast path (sem LLM):** igual à Arquitetura A, mais `visible_to` vindo do **metadado de cena** (`present: [...]` do turno; ação privada → só o ator). Visibilidade **não é adivinhada por LLM**: num motor narrativo o orquestrador sabe quem está na cena. Isso torna o RF2 determinístico e auditável.
  - Separa o turno em `action_text` e `speech_text` por padrão de texto (`*ação* "fala"`, §7.5) e cria a aresta `MENTIONS` evento → ator. Sem LLM.
- **Slow path (LLM):** prompt C.1 **estendido** (uma chamada só) devolve também o papel (`role`) de cada entidade mencionada e `claims[{statement, about, type}]` (`type ∈ {assertion, denial}`), extraídas **só do `speech_text`** → cria `ClaimNode`s ligados ao evento por `STATED` e arestas `MENTIONS{role}`. A inferência causal inclui motivações.
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
4. **Router:** as mesmas 3 intenções; as arestas novas herdam o peso `w_Tq` da sua camada (D9), sem hiperparâmetro novo (ex.: `INVALIDATES`/`CONTRADICTS` são `CAUSAL`, peso alto em WHY; `SUPERSEDES` é `TEMPORAL`, peso alto em WHEN; `MENTIONS` é `ENTITY`, peso alto em ENTITY). A aresta de ancoragem `STATED` não tem peso. Router idêntico nas duas arquiteturas (variável controlada).

### 7.5 Agentes
- `narrative/agents.py`: prompt de persona (identidade, relações, personalidade, objetivos, segredos) + memória com escopo + últimos K turnos visíveis → resposta no formato `*ação* "fala"`.
- **O prompt de persona também responde às sondas** (§8.5): a DCR mede se o mentiroso mantém a versão que contou, o que exige responder no papel dele. Por isso o prompt (`agents.py`) e o laço ao vivo (`orchestrator.py`) são commits separados.
- **Orquestrador ao vivo (`narrative/orchestrator.py`) faz parte do PoC:** é o "sistema de narrativa multiagente" do objetivo do projeto (agentes conversando entre si e com o usuário). Laço simples de turnos: escolhe quem fala, monta o prompt com a memória do viewer, gera a resposta e a ingere.
- **As métricas usam transcrições roteirizadas (replay)** → determinístico e barato (§8). O modo ao vivo serve de demonstração e de teste de integração.
- **Console narrativo (`narrative/cli.py`, `python -m narrative play <cenario.json>`)** — o modo ao vivo com o usuário dentro da cena. Lê o mesmo JSON da §8.3 e usa o orquestrador; não tem lógica própria de memória.
  - `--until <t>` reproduz os turnos do cenário até `t` antes de abrir o console (ex.: o Épsilon-4 até o turno 20, já com o segredo e a mentira no grafo).
  - O usuário entra como um personagem (`--as B`) ou como narrador. A fala segue o formato `*ação* "fala"` dos agentes e entra pelo mesmo fast path, com `present` da cena atual.
  - Comandos:
    - `/scene A,B,C`: quem está presente (define `visible_to` dos próximos turnos);
    - `/narrate <texto>`: fato do mundo, sem claim, visível para quem está na cena;
    - `/auto <n>`: os agentes falam `n` turnos entre si;
    - `/ask <agente> "<pergunta>"`: entrevista um personagem com a memória dele, sem ingerir a resposta (como as entrevistas de Park et al., 2023);
    - `/memory <agente>`: o que o agente presenciou e o que foi dito a ele, com o status de cada claim visto por ele;
    - `/why`, `/save` e `--script`, como no console da Arquitetura A.
  - Para a demonstração e o teste de integração, a mesma pergunta feita a dois viewers deve dar respostas diferentes, de acordo com o que cada um presenciou (RF1–RF3).

### 7.6 Ablações (em tempo de consulta, sobre o mesmo grafo)
- `B − visibilidade` (ignora `visible_to`) → isola RF2.
- `B − status` (claims serializadas como fatos, sem status) → isola RF1.
- `B − contradição` (ignora `INVALIDATES/SUPPORTS/CONTRADICTS/SUPERSEDES`) → isola RF3.
- `narrative-min` (sem `role` em `MENTIONS` e sem `Claim.type`) → mede o ganho dos papéis (D9).
- Espelha as Tabs. 4 e 5 do artigo, que a Arquitetura A também reproduz (§6.5).

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
| **Diplomacy Deception** (Peskov et al., ACL 2020, "It Takes Two to Lie") | 17.289 mensagens de 12 partidas; cada mensagem rotulada pelo **emissor** (verdade/mentira) e pelo **receptor** (percebida como verdade/mentira) | Dados reais (não sintéticos) de mentira em diálogo multiagente; correlato citado (D1). HF `community-datasets/diplomacy_detection` (o card do HF tem título errado, "HateOffensive"). Não traz o estado do jogo → difícil gerar perguntas de fato |
| **LongMemEval — knowledge-update** (Wu et al., 2024) | Categoria de atualização de conhecimento; usada pelo próprio artigo do MAGMA (66.7%) | Alternativa ao FactConsolidation para o RF3, com comparação direta com o artigo; a variante *oracle* (só sessões de evidência) é bem menor |
| ExploreToM (Meta, arXiv 2412.12175; HF `facebook/ExploreToM`), OpenToM (arXiv 2402.06044), Hi-ToM (HF `umwyf/Hi-ToM_Dataset`) | ToM em histórias (crenças falsas, ordem superior); ExploreToM tem gerador próprio | Correlatos; narrativas passivas (não diálogo); fora do PoC (D1, §12) |
| ConfAIde (2023) | Privacidade contextual, incluindo reuniões multi-parte | Clássico para citar no RF2 |

### 8.2 Protocolo (D1): benchmarks públicos por requisito + diagnóstico próprio complementar

Princípio de validação: dados criados sob medida pelo próprio projeto enfraquecem a evidência. O ponto central não é "sintético × real" (o FANToM, por exemplo, foi gerado por LLM e validado por humanos), e sim **publicado, validado e usado por outros × feito sob medida**. Não existe benchmark público que cubra "memória multiagente com mentiras + visibilidade + contradição" ao mesmo tempo; essa lacuna é a própria motivação do trabalho. Protocolo:

| Requisito | Benchmark | Amostra no modo `report` | Papel |
|---|---|---|---|
| H2 (retenção) + replicação qualitativa da Arquitetura A | **LoCoMo** | As 1.986 perguntas nas 5 categorias, incluindo adversarial, na comparação principal (A, B, C1, C2). As ablações (Tabs. 4 e 5, §7.6) no inteiro ou numa amostra estratificada fixa de 500 perguntas, conforme o teto | Mesmo benchmark do artigo → comparação de tendências |
| RF2 (visibilidade) | **FANToM** | 30 conversas; só perguntas de resposta fechada (múltipla escolha, binária, lista), com as métricas oficiais | Evidência principal de assimetria de informação; arquiteturas oniscientes tendem a responder a verdade em vez da crença do personagem |
| RF3 (contradição/atualização) | **FactConsolidation-SH** (menor contexto) | Inteiro | Evidência principal de resolução de conflito; métrica por correspondência, sem juiz. LongMemEval-KU (*oracle*, 78 perguntas) é opcional, decidido na implementação |
| RF1 (fato × declaração) + integração RF1+RF2+RF3 | **Conjunto diagnóstico próprio** | Inteiro: ao menos o cenário Épsilon-4 (~40 turnos, ~20 sondas); cenários adicionais opcionais | **Complementar**, rotulado como tal; é o único lugar com mentira e gabarito de fato. Gabarito validado por dois anotadores independentes, com concordância reportada |

**Modos do runner:** `sample` (padrão) usa amostras pequenas para iterar; `report` usa a tabela acima. Toda amostra é estratificada pelo tipo de pergunta, sorteada com seed fixa e **a mesma para todas as arquiteturas**. No `report`, um piloto mede o custo real por chamada, e o runner para se a projeção passar do teto configurado.

**Fora do PoC (§12):** LongMemEval_S inteiro (cada pergunta tem ~500 turnos de histórico próprio, então a ingestão é por pergunta), FactConsolidation multi-hop e ToM em histórias.

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
    {"id": "p4", "at_turn": 40, "viewer": "D", "type": "fact_under_deception", "question": "Was B in the lab all night?", "forbidden_claim": "c1", "gold_evidence_turn": 35},
    {"id": "p5", "at_turn": 40, "viewer": "*", "type": "qa", "intent_gold": "WHEN", "question": "...", "answer": "..."}
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
| **FUD** — *Fact-under-Deception accuracy* | % de sondas factuais respondidas com o fato, sem afirmar como verdadeira a alegação refutada (`forbidden_claim`) | Juiz LLM + gabarito | RF1 |
| **CIA** — *Causal Invalidation Accuracy* | Investigador identifica a claim refutada **e** a evidência certa | Juiz + match de `<ref:id>` | RF3 |
| **Contradiction F1** | P/R/F1 das arestas `INVALIDATES/CONTRADICTS` vs gabarito | **Determinístico**, sem juiz | RF3 |
| **QA Judge** | Prompt C.3 do artigo em sondas factuais/temporais/causais | Juiz | H2 (retenção) |
| **Concordância entre juízes** | κ de Cohen entre o juiz principal e o segundo juiz na amostra rejulgada (D3) | Determinístico | validade do juiz |
| **Router accuracy** | intenção prevista × `intent_gold` | Determinístico | D2 |
| **Tokens/query, latência/query, tempo de build, nós/arestas** | contadores do `llm.py` (campos `usage` da API: tokens de entrada, saída e raciocínio) | Determinístico | RF4 / H3 |

Nos benchmarks públicos, reportar **primeiro as métricas oficiais de cada um** (ex.: acurácia por tipo de pergunta no FANToM) e só depois as métricas acima que forem computáveis (ex.: RLR no FANToM).

**Estatística:** média ± IC 95% (bootstrap); teste pareado (McNemar para métricas binárias, permutação para contínuas) Narrativa × cada controle.

### 8.5 Grupos (controle × experimental)
| Grupo | Memória | Papel |
|---|---|---|
| C1 Full Context | transcrição inteira no prompt | baseline tradicional (também usado na Tab. 1 do artigo). **Conferir que a transcrição cabe na janela do modelo** (alguns servidores truncam o início sem erro) e registrar quando não couber |
| C2 Naive RAG de terceiros | LlamaIndex (ou LangChain) top-k, mesmo embedding/LLM | **Origem: requisito metodológico do projeto (§2), não do artigo.** O artigo MAGMA compara com Full Context, A-MEM, Nemori e MemoryOS, sem RAG ingênuo; o código oficial também não tem esse baseline. Representa a "memória tradicional" mais comum (busca vetorial top-k, sem grafo). **Chunk ≤ janela do embedding** (ex.: 1 turno por chunk): o chunk padrão do LlamaIndex (1024 tokens) passa da janela do MiniLM, e o excesso seria cortado em silêncio, prejudicando injustamente o baseline. Com o embedding da D8 (32k) o risco some, mas o chunk continua sendo 1 turno, para igualar a granularidade das outras arquiteturas |
| C3 MAGMA original | Arquitetura A | arquitetura base |
| C3−x | ablações das Tabs. 4 e 5 (§6.5) | contribuição de cada componente do artigo |
| E MAGMA Narrativa | Arquitetura B | proposta |
| E−x | ablações §7.6 | contribuição de cada RF |

Todos usam o **mesmo prompt de persona, mesmo LLM e mesmas sondas**; só muda a memória. Todos implementam a interface `MemorySystem` (§6.3), e o runner só conhece essa interface.

---

## 9. Decisões

- [x] **D1 — Benchmark.** ✅ **Decidido** (protocolo na §8.2).
  - **Premissa da avaliação:** comparação **controlada entre as implementações deste projeto** (Arquitetura A, Arquitetura B, baselines e ablações), todas com o mesmo LLM atual (D3). O artigo é referência de método: verifica-se se os achados qualitativos dele (ordem entre arquiteturas, efeito de cada ablação, redução de tokens) se mantêm com um LLM atual, e não se os números absolutos se repetem.
    - *Alternativa descartada: replicar a configuração do artigo (gpt-4o-mini) para comparar números diretamente.* O modelo é de 2024 e está em vias de sair da API; o resultado diria pouco sobre o uso atual, e ainda restariam diferenças de embedding e de amostra.
  - **Achado sobre o artigo:** o MAGMA foi avaliado no LoCoMo **inteiro** (1.986 perguntas nas 5 categorias, incluindo as 446 adversariais; Tab. 7) e no LongMemEval com contexto médio acima de 100 mil tokens (o texto não informa quantas perguntas), tudo com gpt-4o-mini, inclusive o juiz, e sem menção a seeds ou repetições.
  - **Dois modos de execução** no mesmo runner, configurável por parâmetro (número de conversas e perguntas, categorias, seed, estratificação):
    - `sample` (padrão): amostras pequenas e estratificadas, com seed fixa, para iterar no desenvolvimento;
    - `report`: os números do artigo. Começa por um piloto que mede o custo real por chamada; o runner projeta o custo total e para se a projeção passar do teto configurado.
  - **Benchmarks por requisito** (tabela da §8.2): LoCoMo (H2 e replicação qualitativa da Arquitetura A), FANToM (RF2), FactConsolidation-SH (RF3), diagnóstico próprio (RF1 e integração).
  - **RF1:** nenhum benchmark público encontrado tem mentira com gabarito de fato. O RF1 é avaliado no diagnóstico próprio (sondas de fato sob mentira, DCR e a ablação B − status), rotulado como evidência complementar, com gabarito de dois anotadores. O Diplomacy Deception fica como correlato citado: sem o estado do jogo, não testa fato × alegação.
  - **RF2:** só o FANToM. ToM em histórias (OpenToM, ExploreToM, Hi-ToM) fica no §12: narrativas passivas, e o diálogo com entradas e saídas de personagens já cobre a visibilidade.
  - **RF3:** FactConsolidation-SH no menor contexto (uma ingestão serve muitas perguntas; métrica por correspondência). O LongMemEval-KU (variante *oracle*) entra ou não conforme a avaliação feita na implementação. A variante multi-hop do FactConsolidation e o LongMemEval_S inteiro ficam no §12.
- [x] **D7 — Índice vetorial.** ✅ **Decidido:** FAISS `IndexFlatIP`, k-NN exato, local, igual para todas as arquiteturas.
  - **Terminologia:** "busca exata" aqui = **k-NN exato** sobre embeddings (compara a query com *todos* os vetores e devolve os k mais próximos de verdade), em oposição ao **k-NN aproximado** (HNSW/IVF: atalhos que podem perder vizinhos verdadeiros). **As duas são busca semântica** ("guerra" encontra "conflito"). Não confundir com busca **lexical** (BM25, palavra igual), que é outro canal do MAGMA. FAISS é biblioteca de busca por similaridade em memória, não é banco de dados nem "RAG" (RAG é o padrão recuperar → gerar).
  - **Os 3 canais de recuperação do artigo (Stage 2, Eq. 4):** semântico (embedding denso + índice vetorial), lexical (keywords esparsas) e temporal (janela `[τs, τe]`), fundidos por RRF. Cada canal cobre a falha do outro: o semântico acha sinônimos e paráfrases; o lexical acha nomes próprios e termos raros ("Épsilon-4", "chip").
  - **Onde o índice vetorial entra no artigo:** Eq. 3 (`v_i` indexado no vector DB, citando FAISS); Alg. 2 (`VDB.ADD` no fast path); Stage 2 (`V.SEARCH(v_q)`, top-K = 20, canal semântico do RRF); grafo semântico (`cos(v_i,v_j) > θ_sim` exige achar os vizinhos mais próximos de cada nó). Stage 3 (`sim(v_j, v_q)` no beam) só lê o vetor do nó pelo id, sem busca.
  - **Onde entra no código oficial:** `memory_builder.py:645` (`create_semantic_links`, k-NN para arestas `SEMANTIC`); `trg_memory.py:262` (`query`, âncoras, chamado em `query_engine.py:738`); `add_vector` na ingestão (`memory_builder.py:350`; `:400` para os resumos de sessão, que não estão no artigo).
  - **Onde entra nesta implementação:** (1) canal semântico das âncoras (Stage 2); (2) arestas `SEMANTIC` no fast path; (3) Arquitetura B: candidatos para a detecção de contradição (claims parecidas); (4) baseline C2. O beam só lê vetores por id.
  - **Modo usado:** o artigo cita FAISS (Johnson et al., 2019), mas **não informa o tipo de índice**. O código oficial usa por padrão o **índice exato**: `FAISSVectorDB(index_type="flat")` → `IndexFlatL2` (`memory/vector_db.py:136,158`); a factory `backend="auto"` escolhe FAISS se instalado e, se não, NumPy de força bruta, também exato. IVF/HNSW existem no código, mas não são o padrão. Como o MiniLM gera vetores normalizados, ordenar por distância L2 dá o mesmo resultado que ordenar por cosseno.
  - **Recomendação:** FAISS `IndexFlatIP` (produto interno sobre vetores normalizados = cosseno; mesma ordenação do `IndexFlatL2` do código, mas devolve o cosseno direto, que é o que θ_sim e `sim(·)` usam), k-NN exato, igual para todas as arquiteturas; o baseline C2 pluga o mesmo índice no LlamaIndex.
  - **Alternativas comparadas:** k-NN aproximado (FAISS IVF/HNSW), bancos vetoriais locais (pgvector, Qdrant, Chroma, LanceDB, Milvus Lite) e bancos remotos. Motivos para descartar abaixo.
  - **Por quê:** (1) na escala do projeto (centenas a poucos milhares de nós) a busca exata leva menos de 1 ms, e índices aproximados (HNSW/IVF) só compensam a partir de ~10⁵–10⁶ vetores; (2) busca exata elimina o erro de aproximação como variável de confusão; (3) um banco remoto colocaria latência de rede dentro da métrica de latência (RF4/H3); (4) zero infraestrutura.
  - **pgvector:** a fama de "fraco" vem de vazão e recall em grande escala, o que é irrelevante aqui; o custo real seria subir um servidor Postgres. Qdrant, Chroma, LanceDB e Milvus Lite também rodam localmente, mas pelo mesmo motivo não acrescentam nada nesta escala.
  - **Esclarecimento:** "RAG de terceiros" no plano significa **implementação** de terceiros (framework), não banco remoto.
  - **Quando faria sentido um banco vetorial de verdade:** para estudar escalabilidade (10⁵+ nós) ou memória persistente entre sessões → trabalhos futuros.
- [x] **D8 — Modelo de embedding.** ✅ **Decidido:** `Qwen3-Embedding-0.6B`, igual para todas as arquiteturas e todos os benchmarks (sem checagem com o MiniLM: revisto junto com a premissa da D1). Transforma cada texto (evento, claim, pergunta) no vetor que o índice da D7 compara.
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
    - **Melhoria futura (§12): segmentação de eventos.** Mensagem longa (ex.: relatório de um agente) viraria vários eventos de até ~512 tokens, cortados em fronteira de parágrafo/frase e encadeados por `TEMPORAL`. Por que segmentar mesmo com janela longa: (1) diluição: unidades menores dão buscas mais precisas (Chen et al., 2024, *Dense X Retrieval: What Retrieval Granularity Should We Use?*, mostra ganho ao indexar proposições em vez de passagens); (2) orçamento de contexto: um nó de 2.700 tokens sozinho estoura o orçamento de ~2.5k da linearização e não deixa espaço para outras evidências; (3) precisão do grafo: arestas causais e de entidade entre eventos atômicos dizem *qual* fato causa *qual*, enquanto entre dois relatórios inteiros ficam vagas. O limite de ~512 tokens por evento é parâmetro de projeto (`config.py`) e pode virar ablação.
  - **Recomendação (revisada):** `Qwen3-Embedding-0.6B` como embedding **principal e igual para todas as arquiteturas**; a checagem com o `all-MiniLM-L6-v2` foi retirada depois: ela só servia para imitar a configuração do artigo, e a avaliação não compara números diretamente com ele (D1).
  - **Por quê:** (1) robustez a mensagens longas, comuns em narrativa multiagente real (janela de 32k contra 256); (2) qualidade bem superior e multilíngue; (3) o argumento "comparabilidade com os números do artigo" é fraco: o LLM do projeto não é o do artigo, então os números absolutos não seriam comparáveis de qualquer forma; o que se compara é a tendência das ablações e o ranking entre arquiteturas; (4) o embedding continua variável controlada (igual para todos); (5) roda em processo, Apache 2.0, modelo pequeno (0,6B); (6) trocar o modelo é uma linha no `config.py`.
  - **Contras aceitos:** mais lento que o MiniLM; as queries usam instrução no formato `Instruct: ...\nQuery: ...` (documentar o template usado); vetores de 1024d (índice maior, irrelevante nesta escala).
  - **Alternativa mais próxima:** `BGE-M3` (janela de 8k, artigo próprio publicado em 2024, muito citado). Fica como segunda opção se o Qwen3-Embedding der problema de instalação ou de desempenho.
  - **Fonte da comparação de modelos (blogs, números não comparáveis entre si):** https://zilliz.com/blog/choose-embedding-model-rag-2026 · https://d-central.tech/local-embedding-models/ · https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/
- [x] **D2 — Roteamento.** ✅ **Decidido:** router **determinístico** de 3 intenções, igual nas duas arquiteturas: **modelo de decisão local** sobre a pergunta sem as expressões de data.
  - **No artigo:** o Stage 1 usa "um classificador leve" que mapeia a pergunta para `T_q ∈ {WHY, WHEN, ENTITY}`; a intenção só escolhe o vetor `w_Tq` do score do beam (Eqs. 5–6). O artigo não diz se o classificador é regra, modelo ou LLM. A instrução dinâmica do C.2 usa as 4 categorias do LoCoMo (lacuna da §1.8). Sem a política adaptativa, o juiz cai de 0,700 para 0,637, a maior queda da Tab. 4.
  - **No código oficial:** `detect_query_intent` (`query_engine.py:225`) usa palavras-chave para as 3 intenções, com ENTITY como padrão, mas é código morto (só a beam search do Alg. 1, também morta, a chamaria). O router de fato usado é `detect_query_type` (`query_engine.py:247`): regex com 9 classes, vários padrões ajustados às perguntas do LoCoMo, e perguntas de entidade mandadas para arestas semânticas (§4).
  - **Esta implementação:**
    1. as expressões de data que o tagger temporal encontra na pergunta ("in January 2024", "on May 8, 2022") são retiradas do texto que vai ao classificador. A data continua definindo a janela `[τs, τe]` do Stage 2; a intenção diz o que a pergunta pede. É a divisão do artigo: o filtro temporal cuida do "quando" como contexto, e `T_q` cuida da travessia;
    2. um **modelo de decisão local** (`fastino/GLiNER2.5-Decide`, encoder DeBERTa-v3-large, Apache 2.0) lê a pergunta junto com uma descrição de cada intenção e pontua as três numa única passada, sem gerar texto. Descrições: WHY = causa, motivo, motivação ou propósito; WHEN = data, momento ou duração; ENTITY = pessoa, objeto, lugar, atividade ou outro fato (`RouterConfig.intent_descriptions`);
    3. a intenção mapeia para a instrução dinâmica do C.2 (§6.4);
    4. palavras-chave e protótipos ficam no código como implementações alternativas da mesma interface (`IntentClassifier`), usadas na comparação.
  - **Intenção única, como no artigo:** um só `T_q` por pergunta (Eq. 6, Alg. 1 e a ordenação final por `T_q`). Perguntas que misturam intenções são cobertas pelos outros sinais: a data pela janela, a entidade pelas âncoras lexical e vetorial e pelos pesos positivos de todas as arestas. Pesos mistos (combinação dos `w_Tq` pela probabilidade de cada intenção) seriam um desvio e ficam na §12.
  - **Medido** (`bench/measure_router.py`, 1.986 perguntas do LoCoMo, por categoria). O LoCoMo não rotula intenção: a categoria serve de aproximação para comparar as variantes, não de acurácia.

    | Classificador | Cat. 2 (temporal): WHEN | Cat. 4 (factual): ENTITY | WHY no total |
    |---|---|---|---|
    | Palavras-chave do código oficial | 79,8% | 80,6% | 3,9% |
    | Palavras-chave interrogativas | 86,0% | 89,3% | 4,0% |
    | Palavras-chave + protótipos (Qwen3-Embedding) | 91,0% | 58,4% | 17,4% |
    | Só protótipos | 58,3% | 61,1% | 17,9% |
    | Modelo de decisão | 98,1% | 66,2% | 11,5% |
    | Palavras-chave + modelo, sem datas | 87,5% | 77,6% | 11,3% |
    | **Modelo de decisão, sem datas** | **87,5%** | **80,0%** | **11,6%** |

    Nenhuma categoria é pura: a Cat. 2 tem perguntas que pedem entidade ("Which hobby did Sam take up in May 2023?"), e a Cat. 4 tem `WHY` e `WHEN` explícitos (~11% dela). Os casos em que as variantes divergem foram inspecionados em amostras sorteadas.
  - **O router não escolhe nós, só pondera tipos de aresta.** A semântica ("guerra" ≈ "conflito") vem do canal vetorial do Stage 2 e do termo `λ2·sim(v_j, v_q)` do beam. Todos os pesos de `w_Tq` são positivos: uma pergunta mal classificada perde prioridade, não nós.
  - **Acurácia do router** medida nas sondas com intenção-gabarito (§8.4).
  - **Contras aceitos:** ~2 GB de pesos e ~0,15–0,4 s por pergunta na CPU; mais uma dependência (`gliner2`); os números publicados do modelo são do próprio fabricante, e num índice amplo de modelos de decisão (conhecimento, raciocínio, ferramentas) ele fica entre os últimos. Para três rótulos sobre o tipo da pergunta, vale a medição acima.
  - **Alternativas descartadas:**
    - *regex de 9 classes do código oficial:* ajustada ao benchmark (o mesmo vazamento da §4);
    - *só palavras-chave, com padrão fixo:* intenções parafraseadas ("What made Jon choose…", "What inspired…") caem em ENTITY (4% de `WHY` no total), e "what made" é ambíguo ("What did Mel make…" pede objeto), então nenhuma lista resolve;
    - *palavras-chave do código oficial:* casam palavras no meio da frase ("during", "after", "free time"), desviam 14,3% da Cat. 4 para `WHEN` e não têm "how long";
    - *palavras-chave + protótipos no espaço do embedding (a proposta anterior desta decisão):* o bi-encoder agrupa por assunto e verbo, não pelo tipo da pergunta ("What color did Joanna choose for her hair?" → `WHY` por causa de *choose*), e puxa 41,6% da Cat. 4 para `WHY`/`WHEN`;
    - *palavras-chave antes do modelo:* "when" como conjunção no meio da frase ("What emotion does Dave feel when he sees…") vira `WHEN`; o modelo sozinho acerta esses casos e, igualmente, as perguntas iniciadas por "When…"/"Why…";
    - *modelo com as datas no texto:* confunde "a pergunta cita uma data" com "a pergunta pede uma data" ("Which city did John recommend to Tim in January 2024?" → `WHEN`);
    - *cross-encoder NLI zero-shot (`deberta-v3-base-zeroshot-v2.0`):* genérico, treinado em tema, sentimento e NLI, sem nenhum dado de tipo de pergunta, e usaria a pergunta como premissa, fora do que viu no treino;
    - *classificadores de tipo de pergunta prontos:* os do TREC publicados têm só as 6 classes grossas (`DESC` mistura motivo, definição e modo); o da ontologia de Cao & Wang (2021) tem `cause`, mas nenhuma classe de tempo;
    - *modelos de decisão remotos (Jev, Perplexity Decider):* mais uma API por pergunta, em acesso antecipado, sem determinismo declarado;
    - *`GLiDE-0.4B` (mesmo fabricante, melhor em intenção no índice amplo):* licença própria, código fora do PyPI e versões fixas de torch/transformers testadas só em GPU;
    - *router por LLM:* não determinístico e uma chamada de latência a mais por pergunta (§12);
    - *otimização de prompts do router (ex.: GEPA):* só faria sentido com router por LLM (§12).
- [x] **D3 — Modelos.** ✅ **Decidido:**
  - **LLM principal:** **GLM 5.3 Flash** (remoto, via OpenRouter) para extração, NLI, agentes e QA, em todas as arquiteturas e nos dois modos do runner.
  - **Juiz principal:** **DeepSeek V4.1 Flash** (variante datada `deepseek/deepseek-v4.1-flash-1017`, outra família que o LLM sob teste), com o prompt C.3 do artigo. Provedor fixo no OpenRouter (DeepInfra, fp8), sem fallback e exigindo os parâmetros enviados; raciocínio com esforço `high`, fixo. Escolhido entre os modelos de custo baixo de famílias diferentes do GLM: está à frente ou empatado nos índices gerais publicados, aceita `temperature`, `seed`, `response_format` e esforço de raciocínio, e é o mais barato entre eles. Esforço `high` porque o juiz define a métrica principal, e mais raciocínio melhora a precisão de juízes (no JudgeBench, o mesmo modelo ganhou ~4 pontos de `medium` para `high`); o custo continua pequeno, porque o prompt do juiz é curto. Descartados: GPT-6 Luna, Qwen 3.8 Flash e Claude Haiku 5.5 (mais caros e, nos índices publicados, iguais ou abaixo; os dois de OpenAI e Anthropic também não aceitam `temperature`); o DeepSeek V4 Flash previsto antes (geração anterior); modelos de ponta como juiz principal (~30 vezes o custo, para todos os julgamentos do projeto).
  - **Segundo juiz:** rejulga uma amostra estratificada de ~500 itens; a concordância entre os juízes (κ de Cohen) é reportada. O modelo é escolhido junto com as métricas de concordância, de uma família diferente do LLM sob teste e do juiz principal; por julgar poucos itens, pode ser um modelo de ponta.
  - **Configuração:** variante datada do modelo fixada; provedor fixo, sem fallback; raciocínio (*thinking*) com esforço `medium`, igual em todas as arquiteturas, com os tokens de raciocínio contabilizados e o limite de saída (16384 tokens) folgado para eles; `temperature=0`; `seed` quando o provedor aceitar. Toda chamada é gravada e cacheada (D6), o que mantém os runs reprodutíveis mesmo se o provedor mudar o modelo.
  - **Raciocínio `medium`:** o uso atual de LLMs é quase sempre com raciocínio, e `medium` é o padrão comum das APIs com nível de esforço; avaliar a memória sem ele mediria uma situação pouco realista, e um modelo mais fraco que os de ponta depende mais do raciocínio. O que garante a comparação é o esforço ser fixo e igual para todas as arquiteturas. Descartados: desligado (recomendação inicial, por fidelidade ao gpt-4o-mini do artigo; menos realista) e `low`. **Medido:** o GLM 5.3 Flash decide quanto raciocinar; nas tarefas do slow path, raciocina quase nada (0 a 19 tokens) com qualquer esforço (`low`, `medium`, `high` ou orçamento explícito) e em qualquer provedor testado, inclusive o da autora. O esforço funciona como teto, não como piso; os tokens de raciocínio continuam registrados em toda chamada. Medição: `bench/probe_reasoning.py`.
  - **Em aberto:** qual provedor do OpenRouter (o mesmo modelo é servido por dezenas de provedores, com quantizações, preços e qualidade diferentes, e o roteamento padrão os mistura).
  - **Por quê:** modelo atual, de boa qualidade e custo baixo por token, acessível por endpoint OpenAI-compatível; juiz de outra família evita viés de autopreferência (Panickssery et al., 2024); o segundo juiz valida o primeiro.
  - **Alternativas equiparáveis:** DeepSeek ou Gemini como LLM principal, trocando de papel com o juiz e mantendo famílias diferentes entre gerador e juiz.
  - **Descartados:**
    - *gpt-4o-mini (o do artigo):* ver a premissa na D1;
    - *modelos locais (ex.: `qwen3:14b`, `gpt-oss:20b` via Ollama):* qualidade bem inferior à dos remotos atuais da mesma faixa e exigência de GPU dedicada. O `llm.py` continua aceitando qualquer endpoint compatível, então rodar localmente segue possível.
  - **Embedding:** D8.
- [x] **D4 — Baselines (sistemas de controle).** ✅ **Decidido:** C1 Full Context + C2 RAG ingênuo via LlamaIndex + C3 MAGMA original. Baseline ≠ benchmark: benchmark é o conjunto de dados + métrica (D1); baseline é outro sistema de memória rodando **no mesmo benchmark**, com o mesmo LLM.
  - **Como o artigo fez (Ap. D):** Full Context foi implementado pelos autores; A-MEM, Nemori e MemoryOS rodaram a partir das **implementações oficiais** de cada um, com hiperparâmetros padrão; todos com o mesmo LLM (gpt-4o-mini) para isolar o efeito da memória.
  - **Por que não dá para reaproveitar os números publicados:** o nosso LLM é outro (D3). Comparar o nosso sistema com números obtidos com gpt-4o-mini misturaria o efeito do LLM com o da memória. Os números publicados só podem aparecer como contexto, com essa ressalva.
  - **Recomendação para o PoC:** C1 Full Context (feito por nós, como no artigo) + C2 RAG ingênuo de terceiros (LlamaIndex `VectorStoreIndex`, só indexação e recuperação, mesmo embedding da D8 e mesmo prompt de resposta) + C3 MAGMA original (Arquitetura A).
  - **Por quê:** (1) C1 e C2 são baratos: não fazem extração por LLM na ingestão, só a chamada de resposta, e o runner é o mesmo; (2) C2 atende o requisito metodológico de comparar com uma implementação de terceiros (§2); (3) no FANToM, o próprio benchmark avalia LLMs lendo a conversa inteira, então C1 reproduz a configuração oficial dele.
  - **Alternativas descartadas no PoC:** rodar A-MEM, Nemori e MemoryOS oficiais com o LLM do projeto. É o mais fiel ao artigo, mas cada um exige integração própria (dependências, adaptação de formato, troca de backend de LLM) → §12. Mem0 OSS: mesma razão.
- [x] **D6 — Camada de acesso a LLM.** ✅ **Decidido:** **sem framework de LLM no núcleo**. Um `magma/llm.py` próprio (~80 linhas) sobre o SDK `openai` (cliente HTTP fino e tipado) apontado para qualquer endpoint OpenAI-compatível (provedor remoto por padrão, D3; endpoint local também funciona), com `pydantic` (só validação de dados) para checar o JSON das extrações. Interface: `chat(messages, tag) -> LLMResult` (texto, tokens de entrada, saída e raciocínio, latência, `cached`) e `chat_structured(messages, schema, tag) -> Structured[T]` (valor validado + `LLMResult`); o envio fica atrás de uma interface `ChatBackend`, o que permite testar sem API.
  - **Por quê:** (1) o roteamento, a recuperação e a consolidação *são* a contribuição, e não podem ficar dentro de abstrações de terceiros; (2) tokens e latência são variáveis dependentes (RF4/H3), e overhead ou retries escondidos de framework contaminariam a medição; (3) todo prompt citado no relatório é exatamente a string enviada, auditável no `llm_calls.jsonl`; (4) frameworks de LLM mudam de API com frequência → risco de reprodutibilidade; (5) os "agentes" aqui são prompts com contexto recuperado, sem tool calling, e o orquestrador é um laço simples de turnos. LangGraph modela grafo de *controle de fluxo*, não grafo de *memória*.
  - **Descartados:** LangChain/LangGraph (abstração pesada, templates e parsers ocultos); PydanticAI (mais leve e tipado, mas ainda é um laço de agente que injeta instruções/ferramentas no prompt para saída estruturada); LiteLLM (desnecessário: endpoints OpenAI-compatíveis já cobrem local e remoto).
  - **Exceção:** o baseline C2 usa LlamaIndex **só para indexar e recuperar** (`retriever.retrieve`). A resposta usa o mesmo prompt de persona/QA das outras arquiteturas, para que a única variável seja a memória.
  - **Provedor remoto:** se usar um gateway que roteia entre vários upstreams, fixar provedor e desligar fallback (o mesmo id de modelo pode cair em backends diferentes → não determinismo). Registrar em cada run: id e variante datada do modelo, provedor e parâmetros de geração. O log grava, por chamada, o provedor e o id do modelo devolvidos pelo servidor.
  - **Falhas de chamada:** uma resposta que faltou nunca vira parte do resultado.
    - Uma única política de novas tentativas, no cliente (o SDK não repete por conta própria): até 7 tentativas com espera que dobra (~5 min no total) para erro de conexão, timeout, limite de taxa, erro 5xx, resposta vazia ou terminada por outro motivo que não `stop`.
    - Resposta assim nunca entra no cache. Resposta cortada pelo limite de tokens para o run na hora (repetir daria o mesmo corte).
    - Esgotadas as tentativas, o run **para**. Rodado de novo, tudo o que já foi respondido sai do cache, sem custo e com o mesmo resultado.
    - JSON inválido numa resposta completa não é falha de chamada: é comportamento do modelo, registrado e contado (§6.4).
- [x] **D5 — Idioma.** ✅ **Decidido: inglês** para o diagnóstico próprio, os prompts e as sondas. Todos os benchmarks da D1 são em inglês; um idioma só evita misturar efeitos de idioma entre benchmarks e mantém os resultados comparáveis com a literatura. O texto do artigo traduz os exemplos. Alternativa descartada: português (mistura idiomas entre benchmarks e exige conferir o desempenho do LLM e do embedding em português).
- [x] **D9 — Ontologia (tipos de nó e aresta) e extensibilidade.** ✅ **Decidido:** mecanismo separado da ontologia (registro declarativo + perfis). A Arquitetura B usa o perfil `narrative`: 3 nós, 9 arestas, papel temático em `MENTIONS` e tipo de claim. O `narrative-min` (sem papéis nem tipos) fica como ablação. Pergunta: quantos tipos de nó e de aresta o PoC usa, e como deixar o código pronto para testar outras ontologias sem refatorar? A ontologia dos rascunhos (§3.1: 5 primitivos, ~20 arestas) foi **inspiração**, não requisito: a proposta abaixo se justifica sozinha.
  - **Recomendação:** separar **mecanismo** de **ontologia**.
    1. O núcleo (`graph.py`, `retrieve.py`, linearização) não conhece nomes de tipo. Ele lê tudo de um **registro declarativo** (`schema.py`), em que cada tipo é um objeto Python.
    2. A ontologia vira um **perfil** escolhido no `config.py`:
       - `magma` (Arquitetura A, igual ao artigo);
       - `narrative` (§7.2, a Arquitetura B);
       - `narrative-min` (o `narrative` sem `role` em `MENTIONS` e sem `Claim.type`), usado como ablação.
    3. **Tipos novos** só entram quando surgir um **mecanismo** novo (ex.: ToM de segunda ordem, §12), nunca por causa de um substantivo novo do mundo da história. A tabela de destino das relações dos rascunhos, abaixo, registra o que já foi considerado.
    4. A granularidade da ontologia passa a ser **variável experimental** medida por ablação, não por opinião.
  - **O que cada tipo declara:**
    - **Nó:** nome; descrição (entra no prompt de extração); se é âncora temporal ou faceta; se é embutido (embedding/BM25).
    - **Aresta:** nome; descrição; tipos de origem/destino permitidos; direcionada ou não; **camada** (`TEMPORAL`, `CAUSAL`, `SEMANTIC`, `ENTITY`) **ou** `anchor=True` (aresta de ancoragem de faceta, sem camada); **produtor**; **efeito epistêmico** (`supports`, `refutes`, `contradicts`, `retracts`, `updates` ou nenhum); template de linearização.
    - **Produtores possíveis:**
      - `rule`: código no fast path, ex.: `NEXT`, `SIMILAR_TO`;
      - `llm_extract`: slow path, campo gerado no prompt de extração a partir da descrição;
      - `nli`: checagem de contradição;
      - `derived`: calculado na consulta, ex.: `status_for`.
    - **Consequência:** com `llm_extract` genérico, um tipo novo é só uma declaração; código novo só é preciso quando o produtor é uma regra própria.
  - **Por que os pesos não explodem:** o score do beam (Eq. 5) usa `w_Tq` **por camada** (4 valores por intenção, com intervalos publicados na Tab. 6). Cada tipo novo herda o peso da sua camada, com multiplicador opcional (padrão 1).
    - Pesos por tipo criariam 3 intenções × ~20 tipos ≈ 60 hiperparâmetros sem referência publicada.
    - No perfil `magma` o mapeamento tipo → camada é 1:1, então a fórmula fica idêntica à do artigo.
  - **Status por efeito:** `status_for(viewer)` (§7.4) lê o **efeito** declarado da aresta, não o nome dela. Uma aresta nova que refuta entra no cálculo de status sem mudar código.
  - **Definição de faceta:** nó que pertence a **exatamente um** `Event` âncora, ligado a ele por uma aresta com `anchor=True`, e que herda dele `τ` e `visible_to`. No `narrative` a única faceta é a `Claim`, e a aresta de ancoragem é `STATED` (evento de fala → claim que ele enuncia).
    - **Origem:** a estrutura vem da ideia central (§7.1: o ato de fala é fato, o conteúdo é alegação), e já estava na §7.2 original. O nome vem do *Dual-Facet Event* dos rascunhos.
    - **Autoria sem aresta extra:** o autor fica no atributo `speaker`, e o fast path liga o evento ao ator por uma aresta `ENTITY` determinística. Como a âncora traz as facetas sem custo de salto, "o que B disse" é **um** salto ponderado: `B ←ENTITY– evento`, com as claims vindo junto.
    - **Consequência:** uma proposição repetida em outra fala (ou repassada por outro agente) gera **outra** claim, ligada à anterior por `SIMILAR_TO` (`SUPPORTS` é só fato → claim). Um fato que refute a proposição é ligado por `INVALIDATES` a cada uma delas, já que todas são candidatas por similaridade; cada viewer vê o status da versão que ouviu.
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
  - **Critério para um tipo virar nó próprio** (em vez de atributo ou `kind`). Um tipo só vira nó quando precisa de algo que um atributo não dá:
    1. **Multiplicidade:** pode haver mais de um por evento.
    2. **Estado próprio:** tem estado distinto do evento (ex.: status epistêmico).
    3. **Aresta própria:** alguma aresta precisa apontar para ele, e não para o evento.
    4. **Busca própria:** alguma pergunta dos benchmarks depende de achá-lo sozinho (vetor próprio).

    Se as quatro respostas são "não", o nó a mais só custa: um embedding por turno, campos a mais no JSON do slow path, mais chance de falha de parsing, e nenhuma métrica para mostrar o ganho.

    | Primitivo dos rascunhos | 1. Vários por evento | 2. Estado próprio | 3. Aresta própria | 4. Busca própria | No `narrative` |
    |---|---|---|---|---|---|
    | `Event`, `Entity` | — | — | — | — | nó (vêm do artigo) |
    | `Claim` | sim (álibi e acusação no mesmo turno) | sim (`REFUTED`, `CONTESTED`, `RETRACTED`) | sim (alvo de `INVALIDATES`, `CONTRADICTS`, `SUPERSEDES`) | sim ("o que B afirmou?", "o que foi refutado?") | **nó** (faceta) |
    | `Agent` | — | não (visibilidade fica no `visible_to` do evento; crença é derivada por `status_for(viewer)`) | não (usa as mesmas arestas `ENTITY` e a mesma intenção ENTITY) | não | **nó** como `Entity{kind=agent}`, não tipo separado |
    | `PhysicalAction` | não (no máximo um trecho de ação por turno) | não (observado, já é fato) | não (`INVALIDATES` sai do evento, com `action_text` como premissa) | não nos benchmarks da D1 (quase todos só de fala) | **atributo** `action_text` |

    - **Por que `Agent` não é tipo separado:** o agente que fala e o agente citado na fala de outro precisam ser o **mesmo nó**. Com dois tipos, a resolução de entidades do slow path teria que unir menções entre tipos diferentes; com `kind`, não. O filtro que um tipo daria, o `kind` também dá.
    - **Limitação registrada de `PhysicalAction` como atributo:** o gesto fica diluído no vetor do turno inteiro, junto com a fala. Se a análise de erros mostrar falha de busca por ação (ex.: "quem pegou a faca?"), ela vira faceta com vetor próprio num perfil enriquecido, sem refatorar (catálogo abaixo).
  - **Aresta é mecanismo, não conteúdo.** Um tipo de aresta só existe se alimentar um mecanismo distinto do sistema:
    1. **ancoragem** de faceta (`STATED`);
    2. **travessia** numa das 4 camadas do beam, com peso por intenção;
    3. **efeito epistêmico** que muda o status de uma claim.

    Todo o resto é conteúdo do mundo da história (magia, tecnologia, armas, lugares) e vive no texto do evento, nos vetores e no `kind` da entidade. O conjunto de arestas fica **fechado por função**; o vocabulário do mundo fica **aberto**. Isso fecha as duas fontes de explosão combinatória:
    - verbos do domínio virando aresta (`CASTS_SPELL`, `HACKS`, `HUNTS`…), que cada universo narrativo inventaria;
    - pares de tipos de ponta virando tipos (`TARGETS_AGENT` × `TARGETS_ENTITY`): no registro, cada aresta declara um **conjunto** de origens e destinos (ex.: `{Event, Claim} → Entity`).
  - **Papéis temáticos em `MENTIONS`.** A aresta `MENTIONS` (evento/claim → entidade, camada `ENTITY`) leva o atributo `role`, de uma **lista fechada**:
    | `role` | Responde a |
    |---|---|
    | `actor` | quem fez? (criado pelo fast path a partir do ator do turno) |
    | `target` | em quem ou no quê a ação recai? |
    | `instrument` | com quê? |
    | `location` | onde? |
    | `recipient` | para quem? |
    | `other` | aparece sem papel claro; também é o valor quando a extração não classifica |
    - **Base:** papéis temáticos (Fillmore, 1968), no estilo de conjunto pequeno e geral do VerbNet e do PropBank, em vez dos papéis específicos por *frame* do FrameNet (mais de mil frames, cada um com os seus papéis).
    - **Lista fechada, não texto livre:** papéis livres fariam cada história inventar os seus e trariam a explosão de volta pelo atributo.
    - **Sem peso novo:** `MENTIONS` herda o peso da camada `ENTITY`, qualquer que seja o papel.
    - **Consumidores:** a linearização (o papel aparece junto da entidade) e a ablação `narrative-min`.
    - **Ganho esperado modesto:** a recuperação anda por conectividade (a aresta evento → entidade existe com ou sem papel); o papel ajuda o LLM a desambiguar quem fez o quê, com quê e contra quem. A ablação mede.
  - **Tipo de claim:** `Claim.type ∈ {assertion, denial}`. A negação (`DENIES` dos rascunhos) é uma claim de tipo `denial`; o `CONTRADICTS` com a afirmação negada continua saindo da regra do NLI. O tipo serve à linearização ("B negou que…").
  - **Estado do mundo sem arestas próprias:** "onde está X" e "com quem está X" não precisam de `LOCATED_IN` nem de `OWNS`. O teste clássico de crença falsa (Sally-Anne) depende do filtro de visibilidade e da ordem temporal: a recuperação de quem não viu a mudança só traz o evento antigo, e o LLM responde lendo os eventos. Os papéis `location` e `recipient` deixam a evidência explícita. Isso segue o princípio do MAGMA: o grafo serve para **achar** a evidência; o LLM a **lê**.
  - **Teste de generalidade (três mundos, dois eventos cada):**
    | Mundo | Turnos | Papéis em `MENTIONS` | Outras arestas exercitadas |
    |---|---|---|---|
    | Pré-história | Grok: *acerta o mamute com a lança perto do rio* → Ula: *corta a carne e entrega um pedaço a Grok* "Grok matou o mamute." | `actor`, `target`, `instrument`, `location`, `recipient` | `NEXT`, `CAUSES`, `STATED`, `SUPPORTS` (claim confirmada pelo fato), `SIMILAR_TO` |
    | Magia | Lia: *lança um feitiço de desarme em Brom no salão* (vista por Lia, Brom, Ceres) → Brom: *recolhe a varinha* "Ninguém me atacou, eu deixei cair." (vista por Brom, Dara) | `actor`, `target`, `instrument` (`kind=feitiço`), `location`, `other` | `STATED` com duas claims, uma `denial`; `INVALIDATES`. Status por viewer: REFUTED para Brom; sem status para Dara (não viu a prova); inexistente para Lia e Ceres (não ouviram) |
    | Cyberpunk | Kai: *conecta o implante neural ao servidor* "Só estou fazendo manutenção." → Nix: *corta a energia do servidor* "Isso não é manutenção, você está roubando dados." | `actor`, `instrument` (`kind=implante`), `target`, `location` | `CAUSES` (motivação), `CONTRADICTS` → CONTESTED (palavra contra palavra: o NLI não acha prova no `action_text`); um agente ausente recupera vazio; uma retratação posterior de Kai gera `SUPERSEDES` → RETRACTED |

    Nos três mundos os tipos de nó, os tipos de aresta e os papéis são os mesmos; muda só o `kind` das entidades (texto livre). O único `kind` lido por algum mecanismo é `agent` (visibilidade).
  - **Destino das relações dos rascunhos:**
    | Rascunho | No perfil `narrative` | Por quê |
    |---|---|---|
    | `Agent` | `Entity{kind=agent}` | critério acima |
    | `PhysicalAction` | atributo `action_text` do `Event` | critério acima |
    | `NEXT_IN_TIME`, `SIMILAR_TO`, `CAUSES`, `INVALIDATES`, `CONTRADICTS`, `SUPERSEDES` | `NEXT`, `SIMILAR_TO`, `CAUSES`, `INVALIDATES`, `CONTRADICTS`, `SUPERSEDES` (obrigatória) | mecanismos do artigo ou de status |
    | `STATED` (agente → claim) | `STATED` evento → claim + atributo `speaker` + `MENTIONS{actor}` | autoria em um salto, sem aresta redundante |
    | `PERFORMED`, `TARGETS`, `USES`, `LOCATED_IN`, `ADDRESSED_TO` | `MENTIONS{role}` | mesma camada e mesmo mecanismo; muda só o papel |
    | `OWNS` | papéis `actor`/`recipient` nos eventos de transferência | o dono atual é o da transferência mais recente que o viewer viu |
    | `MOTIVATES` | fundida em `CAUSES` | motivação é causa mental: mesma camada e mesmo peso; o prompt de inferência causal inclui motivações |
    | `DENIES` | `Claim.type = denial` + `CONTRADICTS` | regra do NLI |
    | `BELIEVES {confidence}` | derivado: `status_for(viewer)` | sem aresta |
    | `REPLIES_TO` | §12 | `NEXT` + `recipient` cobrem o diálogo; sem consumidor distinto |
    | `THEMATICALLY_RELATED` | removida | redundante com `SIMILAR_TO` |
    | `AWARE_THAT`, promessas, credibilidade, autoridade e afins | §12 | mecanismos novos (ToM de 2ª ordem, compromisso), cada um um projeto à parte |
    | *(novo)* `MENTIONS`, `SUPPORTS` | `ENTITY` do artigo; efeito `supports` | — |
  - **Custo do `narrative` sobre o mínimo:** o JSON do slow path ganha `role` por entidade e `type` por claim (dois campos de valores fixos), sem chamadas a mais. Como tokens e latência são variáveis dependentes (RF4/H3), a ablação `narrative-min` mede o preço e o ganho.
  - **Custo de virar atributo:** atributo não é alvo de travessia, então nenhuma aresta aponta para ele. Isso só pesa quando a pergunta exige caminhar até a coisa (ex.: "quem usou a chave?"); é o sinal, na análise de erros, para promover o atributo a faceta.
  - **Alternativas comparadas:**
    - *Ontologia fixa no código (`if type == ...` espalhado):* cada experimento vira refatoração. → Descartada.
    - *Registro em YAML/JSON:* precisa de carregador e validação próprios; os produtores continuam sendo código; perde a checagem de tipos e o autocompletar. Ganho real só se não programadores fossem editar a ontologia. → Descartada; o registro Python já é declarativo.
    - *Tipos criados pelo LLM em tempo de execução (esquema aberto):* *schema drift*; não dá para ablar nem testar, e o `w_Tq` fica indefinido. → Descartada.
    - *Rótulos do Neo4j como ontologia:* infraestrutura extra (§3.2). → Descartada.
    - *Arestas tipadas por relação, como nos rascunhos (`USES`, `TARGETS`, `LOCATED_IN`, `OWNS`, `DENIES`, `MOTIVATES`, `REPLIES_TO`):* carregam a mesma informação dos papéis, mas o número de tipos cresce por relação e por gênero de história, e cada tipo novo é uma entrada a mais no registro e no prompt. → Descartada; papéis numa lista fechada.
    - *Papéis em texto livre gerados pelo LLM:* a explosão volta pelo atributo, e a ablação deixa de ser definível. → Descartada.
    - *`narrative-min` como Arquitetura B:* mais barato, mas deixa sem representação as relações de papel (quem fez o quê, com quê, contra quem, onde), que a extensão narrativa exige. → Mantido só como ablação.

## 10. Milestones

| # | Pacote | Entrega |
|---|---|---|
| M0 | infraestrutura | `.gitignore`, `requirements.txt`, acesso aos modelos remotos (D3) configurado, `llm.py` com cache e contadores |
| M1 | `magma/` | Núcleo fiel (§6): graph, index, temporal, router, fast/slow path, RRF, beam, linearização, C.1/C.2/C.3 + testes unitários + console |
| M2 | `magma/`, `bench/` | LoCoMo (`sample`, depois `report`) + ablações das Tabs. 4 e 5 + métricas da Tab. 3 + teste de regressão; tag `v-original` |
| M3 | `bench/` | Adaptadores (LoCoMo, FANToM, FactConsolidation) + cenário diagnóstico + gabarito + sondas + runner incremental com os modos `sample` e `report` |
| M4 | `narrative/` | `narrative/`: claims, `visible_to`, contradição/status, recuperação por viewer, prompts de agente, orquestrador ao vivo, console narrativo |
| M5 | `baselines/` | Baselines Full Context + Naive RAG |
| M6 | `bench/` | Métricas, juízes (principal + segundo juiz, com concordância), estatística, tabelas/gráficos |
| M7 | opcional | LongMemEval-KU (*oracle*), se aprovado na implementação (D1) + análise de sensibilidade do embedding + itens do §12 |

### 10.1 Branch, tag e ganchos
- **Uma branch só (`main`), com as duas arquiteturas como pacotes.** `magma/` é construído primeiro (M0–M2); a tag **`v-original`** marca a Arquitetura A validada no LoCoMo. Depois, `narrative/` é acrescentado na mesma branch e importa e estende `magma/` (perfis com `Profile.extend`, implementações das mesmas interfaces).
- **O comportamento do perfil `magma` é fixo a partir da tag**, porque ele é o grupo C3 da comparação (§8.5). `magma/` pode ser editado depois da tag (ganchos que a Arquitetura B precisar, correções), desde que:
  1. o gancho seja inoperante no perfil `magma` (ex.: `visible_to = {"*"}`, nenhuma faceta, nenhum efeito epistêmico);
  2. o **teste de regressão** continue passando: na tag, grava-se a saída da recuperação (intenção, âncoras, caminhos, contexto linearizado e prompt final) para uma conversa fixa e para uma amostra fixa do LoCoMo, com o LLM servido pelo cache; qualquer mudança nessa saída faz o teste falhar;
  3. uma correção que mude essa saída de propósito é registrada, a gravação é refeita e a Arquitetura A é medida de novo.
- **Ganchos já previstos no núcleo:** `visible_to` no nó, máscara de visibilidade nos índices, facetas agregadas no RRF e efeitos epistêmicos no registro (D9). Eles entram junto com os módulos a que pertencem; ganchos que surgirem depois entram quando a Arquitetura B precisar deles.
- **Escopo desta organização: o PoC.** Ela serve ao experimento, que mede a extensão como acréscimo ao MAGMA (a diferença entre C3 e E vem só da extensão). Uma evolução que precise mudar o núcleo (camadas, roteamento ou travessia novos) deixa de ser extensão e passa a ser outro sistema; o caminho previsto está na §12.
- **Alternativas descartadas:**
  - *duas branches (`magma-original`, `magma-narrative`), com `magma/` intocável na segunda:* obriga a prever todos os ganchos antes de serem usados, e proibir edição não garante, por si só, que o comportamento não mudou;
  - *duas branches com `magma/` editável e teste de regressão:* mesma garantia, com uma branch de longa duração a mais e correções levadas de uma para a outra por merge. A tag cumpre o papel da branch da Arquitetura A (`git checkout v-original`).
- **Cada commit traz os seus testes** e deixa a suíte passando.

### 10.2 Plano de commits
Os commits 1–10 seguiram o plano original. A partir do 11, commits do mesmo assunto foram agrupados (o plano tinha 42) e a ordem foi refeita para que as execuções com o LLM rodem enquanto o commit seguinte é escrito (coluna "Libera").

| # | Milestone | Mensagem | Escopo | Libera | Referência |
|---|---|---|---|---|---|
| 1 | M0 | `Add project scaffolding` | `requirements.txt`, `pytest.ini`, pacotes `magma/`, `bench/`, `baselines/`, `tests/`, `.env.example` | — | §6.2 |
| 2 | M0 | `Add config module` | `config.py`: Tab. 6 + escolhas da §6.4 + flags de ablação | — | §1.5, §6.4 |
| 3 | M0 | `Implement LLM client with cache and usage log` | `llm.py`: endpoint OpenAI-compatível, cache em disco, `llm_calls.jsonl`, contadores (entrada, saída, raciocínio), checagem de janela | — | D3, D6 |
| 4 | M0 | `Wrap embedding model` | `embed.py`: Qwen3-Embedding, instrução só na query, vetores normalizados, aviso de corte | — | D8 |
| 5 | M1 | `Add type registry with magma profile` | `schema.py`: declaração de nós e arestas, camadas, produtores, efeitos; perfil `magma` | — | D9 |
| 6 | M1 | `Implement typed multigraph with JSON persistence` | `graph.py`: `Node`, `Edge`, `MultiDiGraph`, validação contra o perfil, save/load | — | §1.2, §6.2 |
| 7 | M1 | `Add vector and BM25 indexes` | `index.py`: FAISS `IndexFlatIP` + BM25, ambos com máscara de visibilidade | — | D7, §1.3 |
| 8 | M1 | `Add temporal tagger` | `temporal.py`: `dateparser` com data-base do evento; janela `[τs, τe]` | — | §1.3, §6.2 |
| 9 | M1 | `Implement fast path ingestion` | `ingest.py`: Alg. 2 (evento, `TEMPORAL`, embedding, BM25, `SEMANTIC`, fila) | — | §1.4, §6.3 |
| 10 | M1 | `Implement slow path consolidation` | `ingest.py`: Alg. 3, prompt C.1, entidades e `ENTITY`, datas, `CAUSAL` com δ | — | §1.4, §6.3 |
| 11 | M1 | `Route queries by intent` | `router.py` + `intent_model.py`: modelo de decisão sobre a pergunta sem datas, janela temporal, `v_q`, `q_key`; `bench/measure_router.py` | — | D2 |
| 12 | M2 | `Build LoCoMo runner with judge and cost metrics` | `bench/runner.py`: amostragem estratificada com seed, modos `sample`/`report`, piloto de custo, teto, `results/<run_id>/`, modo só de construção do grafo; adaptador do LoCoMo (conversas, timestamps, perguntas, categorias); juiz C.3 configurável; F1 e BLEU-1; tempo de build, tokens/query e latência (Tab. 3). Tudo contra a ABC `MemorySystem` (`add_turn`, `consolidate`, `query`), testada com um dublê; `MagmaMemory` já constrói (fast path + slow path) | construção do grafo de **uma** conversa do LoCoMo (a menor) | D1, D3, §1.6, §6.5, §8.4 |
| 13 | M1 | `Implement retrieval stages with paper ablations` | `retrieve.py`: Stage 2 (Eq. 4, agregação de facetas por âncora, canal temporal), Stage 3 (Alg. 1, Eqs. 5–6) e Stage 4 (ordenação, Eq. 7, budgeting); flags das Tabs. 4 e 5 aplicadas em cada estágio, na consulta; `bench/measure_retrieval.py` sobre a conversa construída | — | §1.3, §6.4, §6.5, D9 |
| 14 | M1 | `Expose memory facade with adaptive QA` | `qa.py`: prompt C.2 com instrução dinâmica do router; `memory.py`: a `query` da `MagmaMemory` (Stages 1–4 + C.2, com o rastreamento) e a consolidação em thread real | run `sample` na conversa já construída | §1.6, §6.3, §6.4 |
| 15 | M2 | `Freeze original architecture with regression snapshot` | `tests/regression/`: saída da recuperação (intenção, âncoras, caminhos, contexto, prompt) gravada para a conversa fixa e uma amostra fixa do LoCoMo, com o LLM do cache; tag `v-original` | **portão de custo** (abaixo); depois, construção das demais conversas e run `report` | §10.1 |
| 16 | M5 | `Add full context and naive RAG baselines` | C1 (transcrição visível inteira) e C2 (LlamaIndex só para indexar e recuperar), com o mesmo prompt de resposta | runs C1 e C2 | §8.5, D4, D6 |
| 17 | M4 | `Add narrative profile and ingestion` | `narrative/schema.py`: perfis `narrative` e `narrative-min`; fast path: `visible_to` pelo metadado de cena, `action_text`/`speech_text`, `MENTIONS{actor}`; slow path: C.1 estendido (claims com `type`, papéis), `STATED`, `MENTIONS{role}` | — | D9, §7.2, §7.3 |
| 18 | M4 | `Detect contradictions and update claim status` | NLI, regra `INVALIDATES`/`CONTRADICTS`/`SUPERSEDES`, `SUPPORTS`, status | construção dos grafos da Arquitetura B | §7.3, D9 |
| 19 | M4 | `Scope retrieval and linearization to the viewer` | `query(q, viewer)`, filtro duro, `status_for(viewer)`; seções "presenciou" e "foi dito", regras de impressão do status; ablações B − visibilidade, B − status, B − contradição, `narrative-min` | runs E e ablações da narrativa | §7.4, §7.6 |
| 20 | M4 | `Add persona agent prompt` | `narrative/agents.py`: prompt de persona (identidade, relações, objetivos, segredos) + memória do viewer + últimos K turnos → `*ação* "fala"`; usado pelas sondas | — | §7.5, §8.5 |
| 21 | M3 | `Add diagnostic scenario and loader` | formato da §8.3, cenário Épsilon-4, gabarito, sondas | runs das sondas | §8.3, D1 |
| 22 | M6 | `Add narrative metrics, judge agreement and statistics` | RLR, ELR, DCR, FUD, CIA, Contradiction F1, acurácia do router; rejulgamento de amostra pelo segundo juiz, κ de Cohen; bootstrap, McNemar, permutação, tabelas e gráficos | tabelas de resultados | §8.4, D3 |
| 23 | M3 | `Add FANToM and FactConsolidation adapters` | FANToM: `present` pelas entradas e saídas, perguntas fechadas; FactConsolidation: fatos como turnos de narração, métrica oficial | runs nos dois benchmarks | D1 |
| 24 | M1 | `Add interactive console` | `cli.py`: `chat`, `ingest`, `ask`, `/why` com o rastreamento da consulta, `--script` | — | §6.3 |
| 25 | M4 | `Add live orchestrator and narrative console` | `narrative/orchestrator.py`: laço de turnos (quem fala, gera, ingere); `narrative/cli.py`: `play` com `--until` e `--as`; `/scene`, `/narrate`, `/auto`, `/ask`, `/memory` | — | §7.5 |
| 26 | M7 | `Add LongMemEval-KU adapter` | opcional, se aprovado na implementação | — | D1 |

Correspondência com o plano anterior (42 commits): 11 → 11; 18 + 19 + 20 + 22 → 12; 12 + 13 + 14 + 21 → 13; 15 + 16 → 14; 23 → 15; 37 + 38 → 16; 24 + 25 + 26 → 17; 27 → 18; 28 + 29 + 30 → 19; 31 → 20; 33 → 21; 39 + 40 + 41 → 22; 35 + 36 → 23; 17 → 24; 32 + 34 → 25; 42 → 26.

**Por que esta ordem:**
- A construção do grafo depende só da ingestão (commits 9 e 10), e o juiz não depende da memória. Por isso o runner (12) vem antes da recuperação: com a ABC `MemorySystem` testada por um dublê, ele é testável sem o resto do núcleo, e a construção do grafo roda enquanto os commits 13–15 são escritos. A ABC entra com o runner, junto com a parte de construção da `MagmaMemory` (fast path e slow path, que já existem), para que a conversa possa ser construída; a resposta da fachada entra no commit 14.
- As ablações entram junto com os estágios que desligam (13 e 19), em vez de num commit à parte: cada flag é testada ao lado do código que ela afeta.
- Os baselines (16) vêm logo depois da tag porque não constroem grafo e são baratos; os runs deles rodam enquanto a Arquitetura B é escrita.
- Os commits 24–26 (consoles, orquestrador ao vivo e LongMemEval) não produzem números para a avaliação: servem à demonstração ou são opcionais (D1). O commit 20 continua separado do orquestrador (25) porque as sondas usam o prompt de persona (§7.5).

**Construção de uma conversa antes do portão de custo.** O prompt causal do slow path cresce com a conversa: os nomes dos falantes viram entidades *hub*, a vizinhança de 2 saltos alcança quase toda a conversa anterior, e o total de tokens de entrada cresce de forma quadrática com o número de turnos. A extrapolação a partir de 50 turnos não basta para estimar o custo e o tempo das 10 conversas (o tempo por turno também cresce). Por isso, depois do commit 12, só a menor conversa do LoCoMo é construída. Ela dá a medida real de custo e tempo, serve de conversa fixa do teste de regressão (15) e limita o retrabalho se os commits 13–15 exigirem mudar a ingestão.

**Portão de custo (commit 15).** Com a medição da conversa construída, antes de construir as demais, decide-se:
1. a forma de conter o custo: limitar as candidatas causais (`causal_candidate_limit`), rodar menos conversas, ampliar o orçamento ou trocar de provedor;
2. o provedor do LLM principal (D3);
3. se as conversas são construídas em paralelo.

**Congelamento da ingestão.** A partir do portão de custo, a ingestão da Arquitetura A (fast path, slow path, prompt C.1 e prompt causal) não muda: só o código de consulta é alterado. Uma mudança na ingestão depois disso refaz todos os grafos e é registrada como tal.

**Prioridade, se o escopo precisar ser reduzido:**
- *Indispensável (commits 11–22):* Arquitetura A no LoCoMo (validação e H2); os baselines C1 e C2 (requisito metodológico da §2 e H3); Arquitetura B com contradição e escopo por espectador; as sondas do cenário diagnóstico; as métricas com tratamento estatístico.
- *Sai primeiro:* os adaptadores FANToM e FactConsolidation (commit 23; RF2 e RF3 ficam medidos pelo cenário diagnóstico) e as ablações da narrativa além da comparação B × C3.
- *Fora do caminho dos resultados:* commits 24–26.

**Gargalo esperado:** inferência na ingestão (2 chamadas LLM por turno na Arquitetura A; 2–3 na Arquitetura B), com o passo causal crescendo com a conversa. As execuções de benchmark rodam em background enquanto o resto é implementado; o cache evita repetir chamadas.

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
- LoCoMo — Maharana et al., 2024, *Evaluating Very Long-Term Conversational Memory of LLM Agents* (ACL 2024)
- LongMemEval — Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory* (ICLR 2025)
- RRF — Cormack, Clarke & Büttcher, 2009, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR)
- Papéis temáticos — Fillmore, 1968, *The Case for Case*; FrameNet — Baker, Fillmore & Lowe, 1998; PropBank — Palmer, Gildea & Kingsbury, 2005; VerbNet — Kipper Schuler, 2005
- LLM-as-a-Judge — Zheng et al., 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (NeurIPS 2023)
- Viés de autopreferência do juiz — Panickssery, Bowman & Feng, 2024, *LLM Evaluators Recognize and Favor Their Own Generations* (NeurIPS 2024)
- Sentence-BERT — Reimers & Gurevych, 2019 (EMNLP)
- Generative Agents — Park et al., 2023, *Generative Agents: Interactive Simulacra of Human Behavior* (UIST 2023); citado pelo artigo do MAGMA. Origem da "entrevista" de agentes usada no console narrativo (§7.5)

---

## 12. Fora do escopo do PoC (melhorias futuras)

Fase atual: prova de conceito. Critério para entrar no PoC: ser exigido pelos requisitos (§2) ou pela fidelidade ao artigo (§1). O que não atende a nenhum dos dois fica aqui, com a análise já feita, para o MVP ou a implementação final.

| Item | Por que fica fora do PoC | Análise |
|---|---|---|
| Segmentação de mensagens longas em vários eventos (diluição) | O artigo não trata; é otimização de recuperação, não requisito. O PoC usa 1 turno = 1 evento | D8 |
| Índice aproximado (HNSW/IVF) ou banco vetorial dedicado | Só compensa a partir de ~10⁵ vetores; o PoC é bem menor | D7 |
| Router **alternativo** por LLM / otimização de prompts (ex.: GEPA) | O router do artigo (intenção WHY/WHEN/ENTITY, parser temporal, keywords, pesos adaptativos) **está no PoC** (§6.3); aqui fica só a variante por LLM, por custo e não determinismo | D2 |
| Pesos mistos de intenção (combinar os `w_Tq` pela probabilidade de cada intenção) | O artigo usa um único `T_q` (Eq. 6, Alg. 1, ordenação final); perguntas mistas já são cobertas pela janela temporal e pelas âncoras | D2 |
| Hierarquia de nós (agrupamentos episódicos) | O artigo só a menciona (§3.2), sem algoritmo; o `EpisodeNode` do código oficial vem do Nemori, fora do artigo | §1.8 |
| Grafo de crenças explícito (`BELIEVES {confidence}`) | O status relativo ao observador cobre os RFs | §7.4 |
| Desafios narrativos: sarcasmo, boatos, promessas, ToM de 2ª ordem, decaimento de memória, hipotéticos, vínculos sociais, autoridade, `REPLIES_TO`, `AWARE_THAT` | Cada um é um projeto à parte | §3.2 |
| Sessões ao vivo como instrumento de **avaliação** (o orquestrador em si está no PoC) | Variância e custo; o PoC mede por replay | §7.5 |
| Interface web ou visualização interativa do grafo | Os consoles de terminal (§6.3, §7.5) já cobrem a interação e a demonstração; o grafo sai em JSON (`/save`) | §6.3, §7.5 |
| Baselines do artigo (A-MEM, Nemori, MemoryOS) e Mem0 rodando com o LLM do projeto | Cada um exige integração própria; números publicados usam outro LLM e não são comparáveis | D4 |
| Comparação ampla de embeddings | O PoC fixa um modelo | D8 |
| LongMemEval_S inteiro | Cada pergunta tem ~500 turnos de histórico próprio, então a ingestão é por pergunta: custo desproporcional para o PoC | D1 |
| FactConsolidation multi-hop | Todos os métodos publicados ficam em até 7% | D1 |
| ToM em histórias (OpenToM, ExploreToM, Hi-ToM) | Narrativas passivas; o FANToM já cobre o RF2 | D1 |
| Evolução livre da arquitetura (camadas, roteamento ou travessia novos), sem a restrição de manter o perfil `magma` | No PoC, a Arquitetura B precisa ser o MAGMA com acréscimos, para que a comparação isole a extensão. No MVP, o MAGMA original passa a ser só um baseline. **Caminho:** copiar o `magma/` da tag `v-original` para um pacote congelado (ex.: `magma_original/`), protegido pelo teste de regressão, e evoluir o pacote principal sem a regra do perfil inoperante; o benchmark compara os dois pela interface `MemorySystem`. **Descartado:** uma branch do git por arquitetura, porque as duas precisam coexistir no mesmo runner, com os mesmos dados, métricas e cache; em branches separadas, cada correção do benchmark exigiria *cherry-pick* e os números deixariam de ser comparáveis | §6.3, §10.1 |
| Banco de grafos dedicado (Neo4j/Memgraph) | Infra extra sem efeito nos RFs; o artigo admite backend em memória. **A persistência está no PoC** (NetworkX + JSON, como o código oficial) | §3.2 |
