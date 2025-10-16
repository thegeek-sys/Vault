---
Class: "[[Programmazione per il Web]]"
Related:
  - "[[REST]]"
---
---
## Analisi e schemi
Utilizzando la sintassi YAML per OpenAPI, definisci e documenta le risorse necessarie nel blocco `components`

### Definizione dello schema “Fountain”
Crea lo schema principale per la risorsa Fountain all’interno di `components`/`schemas`

Requisiti:
1. la risorsa deve avere un campo `id`, di tipo interno, `readOnly` e con un esempio valido
2. un campo `state` che accetta solo due valori: `good` o `faulty` (utilizzando enum)
3. i campi `latitude` e `longitude` entrambi di tipo number con formato float (applica )