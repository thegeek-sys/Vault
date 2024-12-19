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