---
Created: 2025-05-08
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
---
## Unicast
Per routing **unicast** si intende la comunicazione tra una sorgente e una destinazione (IP sorgente → IP destinazione)

![[Pasted image 20250508122047.png|550]]

---
## Broadcast
Si parla di routing **broadcast** quando si vuole inviare un pacchetto da un nodo sorgente a tutti i nodi della rete. Ovvero una comunicazione $1$ a $N$, dove $N$ sono tutti i nodi della rete (IP sorgente → indirizzo broadcast)

Esistono due possibilità per eseguire il broadcast:
- **uncontrolled flooding**
- **controlled flooding**
	- sequence number
	- reverse path forwarding

### Uncontrolled flooding
Con l’uncontrolled flooding quando un nodo riceve un pacchetto broadcast, lo duplica e lo invia a tutti i nodi vicini (eccetto a quello da cui lo ha ricevuto)

![[Pasted image 20250508122451.png|150]]

In questo caso se il grafo ha cicli, una o più copie del pacchetto cicleranno all’infinito nella rete

### Controlled flooding
#### Sequence number
Qui non vengono forwardati i pacchetti già ricevuti e inoltrati

In particolare gni nodo tiene una lista di $(\text{indirizzo IP}, \#seq)$ dei pacchetti già ricevuti, duplicati, inoltrati. Quando riceve un pacchetto controlla nella lista, se già inoltrato lo scarta, altrimenti lo forwarda

#### Reverse path forwarding (RPF)
Forwarda il pacchetto se e solo se è arrivato dal link che è sul suo shortest path (unicast) verso la sorgente

![[Pasted image 20250508122803.png|350]]

