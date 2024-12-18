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
- **durabilità** → detta anche persistenza, si riferisce al fatto che una volta che una transazione abbia richiesto un *commit work*, i cambiamenti apportati **non dovranno più essere persi**. Per evitare che nel lasso di tempo fra il momento in cui la base di dati si impegna a scrivere le modifiche e quelli in cui li scrive effettivamente si verifichino perdite di dati dovuti a malfunzionamenti, vengono tenuti dei registri di log dove sono annotate tutte le operazioni sul DB

---
## Schedule di un insieme di transazioni
Per **schedule** si intende un insieme di $T$ transizioni nella cui esecuzione viene mantenuto l’ordine delle singole operazioni di una transizioni, ma ci può essere interleaving tra le transizioni (esecuzione di una parte di una transizione, e una parte di un’altra transizione)

### Schedule seriale
Si parla di **schedule seriale**, quando lo schedule è ottenuto permutando le transazioni in $T$, quindi uno schedule seriale corrisponde ad una esecuzione **sequenziale** (non interfoglia) delle transazioni

---
## Problemi
Consideriamo le due transazioni:
![[Screenshot 2024-12-18 alle 21.22.35.png|300]]
Si possono presentare tre diversi tipi di problemi a causa dell’interleaving

### Aggiornamento perso (update loss)
Consideriamo il seguente schedule di $T_{1}$ e $T_{2}$
![[Pasted image 20241218212457.png|200]]

Se il valore iniziale di $X$ è $X_{0}$ al termine dell’esecuzione dello schedule il valore $X$ è $X_{0}+M$ invece di $X_{0}-N+M$
L’**aggiornamento** di $X$ prodotto da $T_{1}$ viene **perso**

### Dato sporco (dirty data)
Consideriamo il seguente schedule di $T_{1}$ e $T_{2}$
![[Pasted image 20241218212811.png|200]]

Se il valore iniziale di $X$ è $X_{0}$ al termine dell’esecuzione dello schedule il valore di $X$ è $X_{0}-N+M$ invece di $X_{0}+M$
Il valore di $X$ letto da $T_{2}$ è un **dato sporco** (temporaneo) in quanto prodotto da una transazione fallita. Per atomicità quindi bisogna pulire i dati e viene fatto attraverso un rollback a cascata

### Aggregato non corretto
Consideriamo il seguente schedule di $T_{1}$ e $T_{2}$
![[Pasted image 20241218213203.png|240]]

Se il valore iniziale di $X$ è $X_{0}$ e il valore iniziale di $Y$ è $Y_{0}$, al termine dell’esecuzione dello schedule il valore di $somma$ è $X_{0}-N+Y_{0}$ invece di $X_{0}+Y_{0}$
Il valore di $somma$ è un **dato aggregato**