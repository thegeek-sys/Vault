---
Created: 2025-05-09
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello link]]"
---
---
## Comunicazioni

>[!info] Alcuni termini utili
>- host e router sono i **nodi** o le **stazioni**
>- i canali di comunicazione che collegano nodi adiacenti lungo un cammino sono i **collegamenti** (link)
>	- collegamenti cablati
>	- collegamenti wireless
>	- LAN
>- le unità di dati scambiate dai protocolli a livello di link sono chiamate **frame**
>
>![[Pasted image 20250509154153.png]]

### Comunicazione a livello applicazione
![[Pasted image 20250509133833.png|550]]
Due utenti che comunicano possono immaginare che tra di essi esissta un canale logico bidirezionale attraverso il quale si possono inviare messaggi nonostante la comunicazione reale avviene attraverso più livelli e più dispositivi, e vari canali fisici

### Comunicazione a livello trasporto
![[Pasted image 20250509134358.png|550]]
A livello trasporto i protocolli di trasporto forniscono la comunicazione logica tra processi applicativi di host differenti. Infatti gli host eseguono i processi come se fossero direttamente connessi (in realtà possono trovarsi agli antipodi del pianeta)

### Comunicazione a livello rete
![[Pasted image 20250509134447.png|550]]
A livello di rete si ha una comunicazione host-to-host

### Comunicazione a livello di collegamento
![[Pasted image 20250509134736.png|550]]
La comunicazione a livello di collegamento è invece **hop-to-hop**. Internet infatti non è altro che una combinazione di reti unite assieme da dispositivi di collegamento (router e switch)

---
## Link

>[!info]
>I protocolli a livello di collegamento si occupano del trasporto di datagrammi lungo un singolo canale di comunicazione

I nodi all’interno di una rete sono fisicamente collegati da un mezzo trasmissivo come un cavo o l’aria
E’ possibile utilizzare:
- l’intera capacità del mezzo → **collegamento punto-punto** (dedicato a due soli dispositivi)
- solo una parte del mezzo → **collegamento broadcast** (il collegamento è condiviso tra varie coppie di dispositivi)

Un datagramma può essere gestito da diversi protocolli su collegamenti diversi (es. un datagramma può essere gestito da Ethernet sul primo collegamento, da PPP sull’ultimo e da un protocollo WAN nel collegamento intermedio)
Anche i servizi erogati dai protocolli del livello di link possono essere diversi (es. non tutti i protocolli forniscono un servizio di consegna affidabile)

---
## Servizi offerti dal livello di collegamento
### Framing
I protocolli incapsulano i datagrammi del livello di rete all’interno di un frame a livello di link, al fine di separare i vari messaggi durante la trasmissione da una sorgente a una destinazione
Per identificare origine e destinatario vengono usati indirizzi **MAC** (diversi rispetto agli indirizzi IP)

### Consegna affidabile
La consegna affidabile è basata su ACK come nel livello di trasporto, ma è considerata non necessaria nei collegamenti che presentano un basso numero di errori sui bit (fibra ottica, cavo coassiale e doppino intrecciato)

E’ spesso utilizzata nei collegamenti soggetti a elevati tassi di errori (es. collegamenti wireless)

