---
Created: 2024-12-19
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
Un **deadlock** si verifica quando ogni transazione in un insieme $T$ è in attesa di ottenere un lock su un item sul quale qualche altra transazione nell’insieme $T$ mantiene un lock, e quindi rimane bloccata, e quindi non rilascia i lock, e quindi può bloccare anche transazioni che non sono in $T$

---
## Soluzioni per il deadlock
Esistono diverse soluzioni per il deadlock in base a che tipo di approccio utilizzano

### Approcci risolutivi
Quando si verifica una situazione di stallo questa viene risolta

Per **verificare** il sussistere di una situazione di stallo si mantiene il **grafo di attesa**:
- nodi → le transazioni
- archi → $T_{i}\longrightarrow T_{j}$ se la transazione $T_{i}$ è in attesa di ottenere un lock su un item sul quale $T_{j}$ mantiene un lock
Se in tale grafo c’è un ciclo si sta verificando una situazione di stallo che coinvolge le transazioni nel ciclo

>[!note]- Esempio
>![[Pasted image 20241219235406.png]]

Per **risolvere** il sussistere di una situazione di stallo viene fatto un *rollback* su una transazione nel ciclo e successivamente viene fatta ripartire
Quando si parla di rollback nello specifico avviene:
1. la transazione è abortita
2. i suoi effetti sulla base di dati vengono annullati ripristinando i valori dei dati precedenti l’inizio della sua esecuzione
3. tutti i lock mantenuti dalla transazione vengono rilasciati

### Approcci preventivi
Si cerca di evitare il verificarsi di situazioni di stallo adottando opportuni protocolli

#### Esempio
Si ordinano gli item e si impone alle transazioni si richiedere i lock necessari seguendo tale ordine. In tal modo non ci possono essere cicli nel grafo di attesa (e quindi non si può verificare un deadlock)

Se per assurdo supponiamo che le transazioni richiedono gli item seguendo l’ordine fissato e nel grafo di attesa c’è un ciclo
![[Screenshot 2024-12-19 alle 23.50.31.png|500]]

---
## Livelock
Si verifica un **livelock** quando una transazione aspetta indefinitivamente che gli venga garantito un lock su un certo item

una sorta di starvation ????

---
## Soluzioni per il livelock
Il problema dell’attesa indefinita, può essere risolto in due modi:
- con una strategia *first came-first served*
- eseguendo le transazioni in base alle loro priorità e aumentando la priorità di una transazione all’aumentare del tempo in cui rimane in attesa

---
## Abort di una transazione
Un abort può avvenire per una delle seguenti cause:
- La transazione esegue un’operazione non corretta (es. divisione per 0, accesso non consentito)
- Lo scheduler rileva un deadlock
- Lo scheduler fa abortire la transazione per garantire la serializzabilità (timestamp)
- Si verifica un malfunzionamento hardware o software

---
## Punto di commit
Il **punto di commit** di una transazione è il punto in cui la transazione:
- ha ottenuto tutti i lock che gli sono necessari
- ha effettuato tutti i calcoli nell’area di lavoro

Vengono quindi esaurite tutte le situazioni che possono portare ad un deadlock, ma comunque i dati prima del commit sono sporchi (possono portare ad un rollback)

### Dati sporchi
Per dati sporchi si intendono i dati scritti da una transazione sulla base di dati prima che abbia raggiunto il punto di commit

---
## Rollback a cascata
Quando una transazione $T$ viene abortita devono essere annullati gli effetti sulla base di dati prodotti:
- da $T$
- da qualsiasi transazione che abbia letto dati sporchi

