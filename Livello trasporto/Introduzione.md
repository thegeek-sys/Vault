---
Created: 2025-03-23
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello trasporto]]"
Completed:
---
---
## Connessione logica a livello trasporto
I protocolli di trasporti forniscono la **comunicazione logica** tra processi applicativi di host differenti. Con la connessione logica gli host eseguono i processi come se fossero direttamente connessi (in realtà possono trovarsi agli antipodi del pianeta)

I protocolli di trasporto vengono eseguiti nei sistemi terminali:
- lato invio → incapsula i messaggi in **segmenti** e li passa al livello di rete
- lato ricezione → decapsula i segmenti in messaggi e li passa al livello di applicazione

![[Pasted image 20250323112619.png|550]]

---
## Relazione tra livello di trasporto e livello di rete
Mentre il livello di rete regola la **comunicazione tra host** (si basa sui servizi del livello di collegamento), il livello di trasporto regola la **comunicazione tra processi** (si basa sui servizi del livello di rete e li potenzia)

![[Pasted image 20250323113733.png]]

>[!example]