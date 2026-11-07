# Regras de código

Vale para todo o código Python do projeto (`magma/`, `narrative/`, `bench/`, `baselines/`, `tests/`), escrito por pessoas ou por agentes. O design do sistema está em `docs/DESIGN.md`.

## 1. Legibilidade primeiro
O projeto precisa ser lido e explicado de ponta a ponta. Entre uma solução curta e esperta e uma mais longa e óbvia, escolha a óbvia.

- Funções curtas, com um propósito; nomes que dizem o que a coisa é (`anchor_scores`, não `s`).
- Fluxo explícito: nada de metaprogramação, `getattr`/`setattr` dinâmicos, decoradores que mudam assinatura, registro automático por importação, `**kwargs` repassados às cegas ou monkey patching.
- Compreensões só quando cabem numa linha legível; senão, um `for`.
- Sem herança profunda. Composição e funções simples resolvem quase tudo.
- Comentários dizem **por quê**, não **o quê**. Nada de citar itens do design (D3, §6.4) no código.
- As equações e algoritmos do artigo ficam reconhecíveis: um passo do algoritmo vira uma função ou um bloco com nome, e as variáveis seguem os nomes do artigo quando ajudam (`w_tq`, `gamma`, `lambda1`).

## 2. Design desacoplado
Trocar a lógica de uma etapa (roteamento, seleção de âncoras, travessia, linearização) não pode exigir mudanças nas outras.

- Cada etapa expõe uma interface pequena (`typing.Protocol`) e recebe suas dependências pelo construtor. Nenhuma etapa importa a implementação concreta de outra.
- Só a fachada (`memory.py`) e os pontos de entrada (consoles, runner) montam as peças concretas.
- Etapas conversam por dataclasses de entrada e saída (ex.: `RouterOutput`, `AnchorSet`), não por estruturas soltas.
- Parâmetros vêm do `config.py`; nenhum número mágico espalhado pelo código.
- Um módulo novo ou uma variante (ex.: outro router) entra como outra implementação da mesma interface, não como `if` dentro da existente.

## 3. Armazenamento atrás de interfaces
O grafo e os índices são acessados só pelas interfaces (`GraphStore`, índices vetorial e lexical). A implementação atual (NetworkX em memória + JSON) precisa poder ser trocada por Neo4j ou outro backend sem tocar em ingestão, recuperação ou benchmark.

- Nenhum código fora da implementação concreta importa `networkx`, `faiss` ou `rank_bm25`.
- A interface fala em termos do domínio (`add_node`, `neighbors(node_id, layer)`), nunca em objetos da biblioteca.
- Os testes das interfaces rodam contra qualquer implementação.

## 4. Tipagem
Todo código é tipado e passa no `mypy` (configuração em `mypy.ini`).

- Dados internos: `@dataclass(frozen=True)` sempre que não precisarem mudar.
- Dados com formato de JSON que cruzam fronteiras (mensagens para o LLM, arquivos de cenário, linhas de log): `TypedDict`.
- Saída de LLM: modelo `pydantic`, que valida o que chega.
- Conjuntos fechados de valores (camadas, intenções, ablações): `Literal` ou `Enum`, nunca `str` solto.
- Interfaces: `Protocol`.
- **Evitar `dict` puro e `Any`.** `dict[str, X]` só quando a chave é de fato um identificador arbitrário (ex.: `node_id → Node`), nunca como registro com campos. `Any` só na borda com uma biblioteca sem tipos, isolado num único lugar e convertido para um tipo do projeto logo em seguida.
- Sem `# type: ignore` sem um comentário explicando o motivo.

## 5. Testes
- Cada módulo tem testes em `tests/`, sem chamadas reais a APIs (use dublês que implementam a mesma interface).
- Testes de algoritmo usam grafos pequenos montados à mão, com o resultado esperado calculado no próprio teste ou num comentário.
- `pytest` e `mypy` passam antes de cada commit.
