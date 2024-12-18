---
Created: 2024-12-18
Class: "[[Basi di dati]]"
Related:
---
---
## Introduction
In sistemi di calcolo con un a sola CPU i programmi sono eseguiti concorrentemente in modo *interleaved* (interfogliato), quindi la CPU può:
- eseguire alcune istruzioni di un programma
- sospendere quel programma
- eseguire istruzioni di altri programmi
- ritornare ad eseguire istruzioni del primo

Questo tipo di esecuzione è detta concorrente e permette un uso efficiente della CPU

### Accesso concorrente alla BD
In un DBMS la principale risorsa a cui tutti i programmi accedono in modo concorrente è la **base di dati**. Se sulla BD vengono effettuate solo letture (la BD non viene mai modificata), l’accesso concorrente non crea problemi. Se sulla BD vengono effettuate anche scritture (la BD viene modificata), l’accesso concorrente può creare problemi e quindi deve essere controllato

---
## Transazione
Una **transazione** è l’esecuzione di una parte di un programma che rappresenta un’unità logica di accesso o modifica del contenuto della base di dati

### Proprietà delle transazioni
Le proprietà logiche delle transazioni si racchiudono sotto l’acronimo **ACID** (Atomicità, Consistenza, Isolamento, Durabilità). Analizziamole nel dettaglio:
- **atomicità** → la transazione è indivisibile nella sua esecuzione e la sua **esecuzione deve essere totale o nulla**, non sono ammesse esecuzioni parziali (se per qualche problema una transizione non va a termine bisogna fare un rollback sui dati)
- **consistenza** → quando una transazione il database si trova in uno stato consistente e quando la transazione termina, il database deve essere in un altro stato consistente, ovvero **non deve violare eventuali vincoli di integrità**, quindi non devono verificarsi contraddizioni tra i dati archiviati nel DB
- **isolamento** → ogni transazione deve essere eseguita **in modo isolato e indipendente** dalle altre transazione, l’eventuale fallimento di una transazione non deve interferire con le altre transazioni in esecuzione (è ammesso che il risultato cambi a causa di diverse operazioni, è un problema quando ci viene restituito un dato che non era quello richiesto)
- **durabilità** → detta anche persistenza, si riferisce al fatto che una volta che una transazione abbia richiesto un *commit work*, i cambiamenti apportati **non dovranno più essere persi**. Per evitare che nel lasso di tempo fra il momento in cui la base di dati si impegna a scrivere 