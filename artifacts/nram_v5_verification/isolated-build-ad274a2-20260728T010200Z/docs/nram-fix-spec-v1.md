> **Když v UI zvolíš `persona-peak`, `persona-psychedelic`, `persona-normal` nebo MoE, NRAM se pravděpodobně vůbec nezapne.**
Proto všechny režimy opisují tentýž dokument jako studenti v poslední lavici.

# 1. Kritická chyba: `persona-*` aliasy obcházejí NRAM
API správně rozpozná model například `persona-peak`, přidá:

```
request.nram = {
    "enabled": True,
    "profile": "peak",
    "intensity": 0.9,
}
```
Potom ale pošle do enginu stále model:

```
persona-peak
```
SGLang engine však zapne NRAM pouze tehdy, když je název modelu v tomto seznamu:

```
NRAM_ENABLED_ALIASES = {
    "nram-gpt-oss-20b",
    "nram-deepseek-r1-qwen-7b",
    "nram-qwen3-14b-awq",
}
```
`persona-peak`, `persona-normal` ani další aliasy v seznamu nejsou.

Rozhodující kontrola je:

```
def _is_nram_enabled(self, request):
    if request.model in NRAM_ENABLED_ALIASES:
        ...
    return False
```

## Výsledek
Při použití persona modelů se neaplikuje:

- altered-state developer instruction,
- příslušný NRAM profil,
- logit processor,
- repetition policy,
- phenomenon steering,
- žádná skutečná odlišnost jednotlivých stavů.
To dokonale vysvětluje, proč `normal`, `microdose`, `psychedelic`, `peak` a `dissociative` produkují velmi podobnou strukturu a drží se původního textu.

### Oprava
Uvnitř `_route_persona()` musíš veřejný alias převést na skutečný NRAM model:

```
async def _route_persona(
    request: ChatCompletionRequest,
    public_model: str,
) -> dict[str, Any]:
    engine = get_sglang_engine()
    if engine is None:
        raise InferenceError(
            "SGLang engine not available",
            code="engine_unavailable",
        )

    profile = _PERSONA_MODEL_MAP[public_model]

    request.model = "nram-qwen3-14b-awq"
    request.nram = {
        "enabled": True,
        "profile": profile,
        **(request.nram or {}),
    }

    engine_request = _to_engine_request(request, request.messages)
    response = await engine.complete(engine_request)

    result = response.model_dump()
    result["model"] = public_model
    return result
```
Veřejně tedy vrátíš `persona-peak`, ale uvnitř jede `nram-qwen3-14b-awq`.

# 2. Stejná chyba ničí MoE
MoE vytvoří subrequest pro jednotlivé profily, nastaví jim `nram`, ale opět ponechá model:

```
nram-moe-orchestrator
```
Ani tento název není v `NRAM_ENABLED_ALIASES`.

Takže aktuální „MoE“ pravděpodobně dělá:

```
baseline Qwen
baseline Qwen
baseline Qwen
baseline Qwen
        ↓
baseline Qwen synthesizer
```
Profily jsou různě pojmenované, ale engine je neaktivuje.

Navíc každý expert dostává jen **100 výstupních tokenů**, výsledek se ořízne na 200 a následně 150 znaků a finální syntéza smí mít maximálně dvě až tři věty.

To není MoE pro vizionářský dokument. Je to speed dating čtyř useknutých odstavců.

### Oprava
Uvnitř `query_persona()`:

```
sub_request.model = "nram-qwen3-14b-awq"
sub_request.nram = {
    "enabled": True,
    "profile": profile,
}
sub_request.max_tokens = 800
sub_request.seed = request.seed
```
A pro dokumentový MoE bych nepoužíval jen čtyři stejné modely s různým „stavem“. Použil bych skutečně jiné role:

- Future Anthropologist
- Interface Radical
- Product Minimalist
- Economic Architect
- Adversarial CTO
- Science-fiction Prototyper
NRAM profil může modulovat způsob jejich práce, ale **rozdílnost musí vzniknout hlavně rozdílným úkolem a rozdílnou perspektivou**.

# 3. V defaultním Docker Compose pravděpodobně neběží logit steering
Engine vytvoří `TokenBiasCompiler` pouze tehdy, když dostane tokenizer:

```
if tokenizer is not None:
    self._bias_compiler = TokenBiasCompiler(tokenizer)
```
A vlastní logit processor vloží pouze pod podmínkou:

```
if nram_enabled and self._bias_compiler is not None:
    ...
    payload["custom_logit_processor"] = ...
```
Tokenizer se načítá z proměnné:

```
NRAM_TOKENIZER_PATH
```
Pokud není nastavena, loader vrátí `None`.

Jenže v `compose.yaml` má tokenizer a `/models` připojený pouze kontejner `sglang`. Kontejner `nram-api`, který sestavuje token policy, nemá:

- `NRAM_TOKENIZER_PATH`,
- mount `/models`,
- ani jinou cestu k tokenizeru.

## Praktický výsledek
Defaultní stack může používat:

- system prompt,
- profile prompt,
- phenomenon text,
ale **ne vlastní token-level biasing**, přestože UI říká „real-time logit steering“.

### Oprava Compose

```
nram-api:
  environment:
    NRAM_ENGINE: sglang
    SGLANG_BASE_URL: http://sglang:30000/v1
    SGLANG_MODEL: nram-qwen3-14b-awq
    NRAM_TOKENIZER_PATH: /models

  volumes:
    - E:\_MODELS\huggingface\hub\models--Qwen--Qwen3-14B-AWQ\snapshots\31c69efc29464b6bb0aee1398b5a7b50a99340c3:/models:ro
    - workflow-data:/app/data
```
A při startu bych failnul aplikaci, když je NRAM aktivní a tokenizer chybí:

```
if sglang_enabled() and tokenizer is None:
    raise RuntimeError(
        "NRAM_ENGINE=sglang requires NRAM_TOKENIZER_PATH. "
        "Refusing to run in silent prompt-only mode."
    )
```
Silent fallback je tady jedovatý. UI tvrdí, že řídí tokeny, ale motor může mít odpojený volant.

# 4. Slider `intensity` se fakticky ignoruje
Streamlit posílá:

```
{
    "profile": "...",
    "intensity": intensity,
    "coherence_floor": coherence,
    "associative_distance": intensity,
}
```
Jenže `NRAMState` žádné obecné pole `intensity` nemá. Má konkrétní dimenze, například:

- `visionary_intensity`,
- `contrarian_force`,
- `associative_distance`,
- `novelty_target`,
- `repetition_penalty`.
Engine přepisuje pouze explicitní pole. `intensity` mezi nimi není.

Takže tento slider je v současnosti převážně **knoflík připojený k dekorativní lampičce**.

### Lepší varianta
Buď ho úplně zrušit, nebo definovat, co intenzita znamená:

```
def apply_global_intensity(
    base: NRAMState,
    intensity: float,
) -> NRAMState:
    neutral = PROFILES["normal"]

    def mix(a: float, b: float) -> float:
        return a + (b - a) * intensity

    return NRAMState(
        visionary_intensity=mix(
            neutral.visionary_intensity,
            base.visionary_intensity,
        ),
        contrarian_force=mix(
            neutral.contrarian_force,
            base.contrarian_force,
        ),
        associative_distance=mix(
            neutral.associative_distance,
            base.associative_distance,
        ),
        theatricality=mix(
            neutral.theatricality,
            base.theatricality,
        ),
        novelty_target=mix(
            neutral.novelty_target,
            base.novelty_target,
        ),
        coherence_floor=mix(
            neutral.coherence_floor,
            base.coherence_floor,
        ),
        # další dimenze...
    )
```

# 5. Developer prompt a logit policy používají různé stavy
V `complete()` se developer instruction vytvoří z čistého profilu:

```
profile = PROFILES.get(profile_name, ...)
policy = compile_policy(profile, ...)
developer_instruction = policy.developer_instruction
```
Až později se v `_build_upstream_payload()` vezme profil znovu a aplikují se request overrides.

To znamená:

```
system prompt        = původní profil
logit processor      = profil + override
UI                   = tvrdí třetí sadu parametrů
```
Máš tři volanty připojené ke dvěma nápravám.

### Správně
Jednou vytvořit výsledný `NRAMState`:

```
state = resolve_request_state(request.nram)
policy = compile_policy(
    state,
    max_tokens=request.max_tokens or 512,
    profile_name=profile_name,
)
```
A stejnou `policy` použít:

- pro developer instruction,
- pro token bias,
- pro telemetry,
- pro audit,
- pro zobrazení v UI.

# 6. Agentní persony nepoužívají deklarované teploty
U `psychedelic_synthesizer` dokumentace tvrdí:

```
temperature=1.05
associative_distance=0.94
```
Ale `BasePersona.build_nram_options()` posílá pouze:

```
{
    "enabled": True,
    "profile": ...,
    "intensity": ...,
    "seed": ...,
}
```
Teplota, associative distance, contrarian force a coherence floor se neposílají.

`invoke_model()` také nepřidá teplotu do `chat_options`.

Výsledkem je default FastAPI teplota `1.0` pro prakticky všechny agentní persony.

To znamená, že Archaeologist, CTO i Dictator nemusí běžet s nízkou deklarovanou teplotou. To zvyšuje:

- halucinace,
- JSON chyby,
- nestabilitu,
- přechod na fallbacky.

### Oprava

```
def build_chat_options(self) -> dict[str, Any]:
    return {
        "temperature": self._nram_config.get("temperature", 0.7),
        "seed": self._workflow_seed,
        "nram": {
            "enabled": True,
            "profile": self._nram_config["profile"],
            "visionary_intensity":
                self._nram_config.get("visionary_intensity"),
            "associative_distance":
                self._nram_config.get("associative_distance"),
            "contrarian_force":
                self._nram_config.get("contrarian_force"),
            "coherence_floor":
                self._nram_config.get("coherence_floor"),
        },
        "include_telemetry": True,
    }
```

# 7. Režim Peak není vizionářský. Je úmyslně rozbitý.
`peak` prompt přímo požaduje:

- heavy fragmentation,
- thought loops,
- dissolution of self,
- rozpad čísel a logiky,
- nízkou koherenci,
- intenzivní mystický zážitek.
To je vhodné pro experiment se stylem textu. Není to vhodné pro:

> „Steve Jobs na LSD, ale pořád technicky a investorsky brilantní.“
Po opravě persona aliasů se může stát, že `peak` přestane kopírovat dokument, ale začne místo toho produkovat rozpadlý kaleidoskop slov. To by bylo technicky správně podle promptu, ale produktově špatně podle tvého cíle.

## Potřebuješ dva oddělené systémy

### Altered-state simulator

- fragmentation,
- looping,
- forgetting,
- dissolution,
- nízká coherence.

### Visionary product cognition

- vysoká asociativní vzdálenost,
- vysoká novelty,
- vysoká contrarian force,
- vysoká product obsession,
- **současně coherence 0.85–0.95**,
- nulové forgetting,
- nulové looping,
- nulové dissolution.
Například nový profil:

```
VISIONARY_PEAK = NRAMState(
    visionary_intensity=0.98,
    contrarian_force=0.92,
    product_obsession=0.97,
    human_focus=0.91,
    rhetorical_compression=0.78,
    associative_distance=0.90,
    theatricality=0.82,
    emotional_voltage=0.76,
    coherence_floor=0.88,
    novelty_target=0.96,
    repetition_penalty=0.65,
    corporate_jargon_penalty=0.95,
)
```

# 8. Logit-level „associative jump“ není sémantická asociace
Aktuální mechanismus `associative_jump` pouze zploští celý logit distribution:

```
row[:] = row * scale
```
A `dissolution` udělá prakticky totéž ještě silněji.

`Synesthesia` přidá malý bias náhodně rozesetým tokenům ve vocabulary.

To vytváří:

- vyšší náhodnost,
- méně stabilní syntax,
- občas nečekané slovo.
Nevytváří to automaticky:

- spojení AI s urbanismem,
- nový obchodní model,
- nový typ rozhraní,
- alternativní produktovou ontologii.

> Logit processor umí zamíchat barvy na paletě. Neumí sám rozhodnout, že místo portrétu namaluje město uvnitř velryby.
Pro vzdálené, ale smysluplné asociace potřebuješ před generováním skutečně dodat vzdálené koncepty:

```
V271
+ operační systémy
+ herní world models
+ biologická paměť
+ pojišťovací trhy
+ distribuovaná identita
```
Potom NRAM může ovlivnit, jak odvážně je model spojí.

# 9. Dokument pořád vstupuje jako hotová próza
Výstupy, které jsi nahrál, opakují stejnou výchozí tezi, stejné pořadí argumentů a často stejné formulace jako původní PDF. Například zdroj i několik variant začínají veřejnými signály z let 2024 až 2026, pokračují agentním runtime a končí `consumer continuity layer`.

To je očekávatelné. Regular Chat posílá modelu jediný user prompt beze změny:

```
"messages": [
    {"role": "user", "content": prompt},
]
```
Pokud prompt obsahuje celý původní odstavec, model ho používá jako nejvýraznější strukturální kotvu.

NRAM mění pravděpodobnost tokenů. **Neodstraní zdrojovou strukturu z kontextu.**

# 10. Agentic Pipeline zatím neumí skutečně ingestovat repozitář
Tohle je druhá velká věc.

`repository_ingestion` pouze vypíše event:

```
"Ingesting repository"
```
`evidence_validation` rovněž pouze vypíše event.

`roadmap_builder` pouze vypíše event.

`final_evidence_audit` pouze vypíše event.

Revision loop obsahuje:

```
# In production: send revise hypotheses back...
# For now, log and continue
```
Archaeologist dostane pouze řetězec:

```
Analyze the repository at D:\...
```
ale nemá:

- filesystem tool,
- GitHub tool,
- shell,
- file reader,
- seznam souborů,
- obsah zdrojových souborů.
Model tedy fyzicky nemůže analyzovat repozitář. Buď si ho vymyslí, nebo spadne na fallback.

Evidence Ledger je dobře navržená SQLite databáze, ale nikde v orchestru není implementován crawler, který by ji skutečnými důkazy naplnil.

A Product Dictator vůbec model nevolá. Vezme prvního přeživšího a vytvoří hardcoded směr a jeden roadmap item.

To znamená, že současná agentní pipeline je zčásti:

- reálná orchestrace,
- reálné persistence,
- reálné model calls u některých person,
- ale také několik scénografických kulis bez implementace za dveřmi.

# 11. Anglický token bias nad českým dokumentem je slabý
Pozitivní a negativní lexémy jsou pouze anglické:

```
human
experience
purpose
simple
create
future
synergy
stakeholder
framework
digital
transformation
```
Tokenizer compiler navíc přijme jen lexém, který je jedním tokenem. Víceslovné a rozdělené výrazy zahodí.

Při českém výstupu tedy tento bias zasahuje jen omezenou část generace.

Přidej jazykové policy:

```
POSITIVE_CONCEPTS_CS = [
    "člověk",
    "zkušenost",
    "smysl",
    "jednoduchost",
    "tvořit",
    "představit",
    "změna",
    "rozhraní",
    "budoucnost",
    "řemeslo",
]

NEGATIVE_CONCEPTS_CS = [
    "synergie",
    "stakeholder",
    "robustní",
    "komplexní",
    "transformace",
    "ekosystém",
]
```
Ještě lepší je generovat lexémy dynamicky podle tématu a jazyka.

# Co je tedy skutečnou příčinou tvých výstupů

## Přímý causal chain

```
Nahraješ původní odstavec
          ↓
Regular Chat pošle celý odstavec Qwenu
          ↓
persona alias nepřepne model na NRAM alias
          ↓
NRAM není aktivní
          ↓
Qwen udělá bezpečnou parafrázi zdroje
          ↓
MoE zopakuje několik baseline odpovědí
          ↓
baseline synthesizer je zkomprimuje
          ↓
výsledkem je původní dokument s jinak učesanými vlasy
```
A i po opravě aliasu:

```
tokenizer v nram-api pravděpodobně chybí
          ↓
žádný custom logit processor
          ↓
zůstává hlavně altered-state system prompt
          ↓
Peak začne fragmentovat text, ale nevytvoří lepší produktovou vizi
```

# Pořadí oprav

## P0, opravit okamžitě

1. V `_route_persona()` interně nastavit `request.model = "nram-qwen3-14b-awq"`.
2. Totéž pro každý MoE subrequest.
3. Připojit tokenizer do `nram-api`.
4. Při chybějícím tokenizeru failnout, ne tiše degradovat.
5. Přidat integrační test, který ověří, že persona alias skutečně vytvoří `custom_logit_processor`.

## P1

1. Sjednotit výpočet výsledného `NRAMState`.
2. Opravit `intensity`.
3. Předávat skutečné persona temperature a overrides.
4. Oddělit Altered-State Simulator od Visionary Product Engine.
5. Přidat české nebo dynamické lexémy.

## P2, nutné pro dokumenty

1. Vytvořit `Document Innovation Workflow`:

```
PDF
 ↓
Evidence Extractor
 ↓
Fact Ledger bez původní prózy
 ↓
Contrarian Deconstructor
 ↓
6 skutečných expertů
 ↓
Concept Tournament
 ↓
Visionary Composer
 ↓
Anti-copy + factuality gate
 ↓
finální dokument
```

1. Composer nesmí dostat původní odstavce, pouze:

- fakta,
- citace,
- schopnosti V271,
- rozpory,
- vybrané nové koncepty.

1. Přidat automatické odmítnutí při vysoké podobnosti se zdrojem.

