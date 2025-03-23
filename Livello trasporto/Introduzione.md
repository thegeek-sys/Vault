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
![[Pasted image 20250323113733.png]]

Mentre il livello di rete regola la **comunicazione tra host** (si basa sui servizi del livello di collegamento), il livello di trasporto regola la **comunicazione tra processi** (si basa sui servizi del livello di rete e li potenzia)

>[!example] Analogia con la posta ordinaria
>Una persona di un condominio invia una lettera a una persona di un altro condominio consegnandola/ricevendola a/da un portiere
>
>In questo caso:
>- i processi sono le persone
>- i messaggi delle applicazioni sono le lettere nelle buste
>- gli host sono i condomini
>- il protocollo di trasporto sono i portieri dei condomini
>- il protocollo del livello di rete è il servizio postale
>
>>[!hint]
>>I portieri svolgono il proprio lavoro localmente, non sono coinvolti nelle tappe intermedie delle lettere (così come il livello di trasporto)

---
## Indirizzamento
La maggior parte dei sistemi operativi è multiutente e multiprocesso (ci sono diversi processi client attivi e diversi processi server attivi)

![[Pasted image 20250323115546.png|center|500]]

Per stabilire una comunicazione tra i due processi è necessario un metodo per individuare:
- host locale → tramite indirizzo IP
- host remoto → tramite indirizzo IP
- processo locale → tramite numero di porta
- processo remoto → tramite numero di porta

>[!info] Indirizzi IP vs. numeri di porta
>![[Pasted image 20250323115740.png]]
>
>L’indirizzo IP + la porta forma il **socket address**

---
## Incapsulamento/decapsulamento
![[Pasted image 20250323115919.png]]

I pacchetto a livello di trasporto sono chiamati **segmenti** (*TCP*) o **datagrammi utente** (*UDP*)
