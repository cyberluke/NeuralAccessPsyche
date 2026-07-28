# Forenzní audit NRAM v5 - Pokročilé techniky

**Datum:** 2026-07-27  
**Audit provedl:** Code Agent  
**Status:** KRITICKÝ NÁLEZ

## Shrnutí

Všechny "pokročilé techniky" implementované ve Sprintu 3 jsou **pouze API kostry bez reálného zapojení do inference pipeline**.

## Nálezy

### 1. ConceptorController
- **Soubor:** `core/steering/conceptor_steering.py`
- **Status:** Implementováno, ale **nepoužíváno**
- **Důkaz:** `grep -r "ConceptorController" core/` vrátí pouze definici třídy, žádné importy

### 2. HiddenStateProbes / ClosedLoopController
- **Soubor:** `core/steering/hidden_state_probes.py`
- **Status:** Implementováno, ale **nepoužíváno**
- **Důkaz:** Žádné importy v runtime kódu

### 3. DExpertsController
- **Soubor:** `core/steering/dexperts.py`
- **Status:** Implementováno, ale **nepoužíváno**
- **Důkaz:** Žádné importy v runtime kódu

### 4. MultiVectorController
- **Soubor:** `core/steering/multi_vector_controller.py`
- **Status:** Implementováno, ale **nepoužíváno**
- **Důkaz:** Žádné importy v runtime kódu

## Co je skutečně funkční

### ✅ Funkční komponenty (verifikováno end-to-end testem)
1. **TokenTrieConstraint** - phrase masking (Layer 2)
2. **EntropyController** - PID servo (Layer 3)
3. **ConceptInjector** - concept injection (Layer 3)
4. **NRAMLogitProcessor** - custom logit processor
5. **ActivationAddition** - vektorová steering (základní)

### ❌ Nefunkční komponenty (API kostry)
1. **ConceptorController** - subspace blending
2. **HiddenStateProbes** - closed-loop monitoring
3. **ClosedLoopController** - adaptive steering
4. **DExpertsController** - expert/anti-expert kombinace
5. **MultiVectorController** - multi-vector combination

## Důsledky

1. **Ablační studie (A-F)** testuje pouze funkční komponenty
2. **Konfigurace D a E** (activation steering) fungují pouze na základní úrovni
3. **Konfigurace F** (full NRAM v5) nezahrnuje pokročilé techniky
4. **Výstupy se liší** díky funkčním komponentám (phrase masking, entropy, concepts)

## Doporučení

### Krátkodobé (před RELEASE)
1. **Odstranit nebo označit jako "experimental"** všechny nefunkční komponenty
2. **Aktualizovat dokumentaci** - neslibovat funkčnost, která neexistuje
3. **Zaměřit se na funkční komponenty** - optimalizovat phrase masking, entropy control, concept injection

### Dlouhodobé (po RELEASE)
1. **Integrovat ConceptorController** do forward hooks
2. **Implementovat HiddenStateProbes** pro closed-loop control
3. **Nastavit DExperts** s reálnými expert modely
4. **Propojit MultiVectorController** s ActivationAddition

## Závěr

NRAM v5 má **silné jádro** (phrase masking, entropy control, concept injection), které prokazatelně mění výstupy. Pokročilé techniky jsou **výzkumné prototypy** bez produkční integrace.

**RELEASE CANDIDATE** status je oprávněný pouze pro funkční komponenty.
