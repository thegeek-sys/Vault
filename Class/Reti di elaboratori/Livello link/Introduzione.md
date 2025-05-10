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

>[!question] Dove è implementato il livello di collegamento?
>In tutti gli host ed è realizzato in un adattatore (*NIC*, **network interface card**) come la scheda Ethernet, PCMCI e 802.11 che implementano il livello di collegamento e fisico
>
>E’ una combinazione di hardware, software e firmware
>![[Pasted image 20250509162723.png|350]]

---
## Servizi offerti dal livello di collegamento
### Framing
I protocolli incapsulano i datagrammi del livello di rete all’interno di un frame a livello di link, al fine di separare i vari messaggi durante la trasmissione da una sorgente a una destinazione
Per identificare origine e destinatario vengono usati indirizzi **MAC** (diversi rispetto agli indirizzi IP)

### Consegna affidabile
La consegna affidabile è basata su ACK come nel livello di trasporto, ma è considerata non necessaria nei collegamenti che presentano un basso numero di errori sui bit (fibra ottica, cavo coassiale e doppino intrecciato)

E’ spesso utilizzata nei collegamenti soggetti a elevati tassi di errori (es. collegamenti wireless)

### Controllo di flusso
Evita che il nodo trasmittente saturi quello ricevente

### Rilevazione degli errori
Gli errori sono causati dalle *interferenze* (attenuazione del segnale e rumore elettromagnetico) e il nodo ricevete può individuarli grazie all’inserimento, da parte del nodo trasmittente, di bit di controllo di errore all’interno del frame

### Correzione degli errori
Il nodo ricevente determina anche in punto in cui si è verificato l’errore e lo corregge

---
## Adattatori
![[Pasted image 20250509162816.png]]

Dal lato del mittente:
- si incapsula un datagramma in un frame
- viene impostato il bit di rilevazione degli errori, trasferimento dati affidabile, controllo di flusso, etc.

Dal lato del ricevente:
- si individuano gli errori, trasferimento dati affidabile, controllo di flusso, etc.
- vengono estratti i datagrammi e passati al nodo ricevente

---
## Due sottolivelli
Il livello di collegamento in ulteriori due sottolivelli:
- **Data-Link Control** (*DLC*)
- **Media Access Control** (*MAC*)

![[Pasted image 20250509163618.png|280]]

Il **Data-Link Control** si occupa di tutte le questioni comuni sia ai collegamenti punto-punto che a quelli broadcast, ovvero:
- framing
- controllo del flusso e degli errori
- rilevamento e correzione degli errori

Si occupa dunque delle procedure per la comunicazione tra due nodi adiacenti (comunicazione nodo-a-nodo), indipendentemente dal fatto che il collegamento sia dedicato o broadcast

![[Pasted image 20250509164921.png|280]]

Il **Media Access Control** si occupa solo degli aspetti specifici dei canali broadcast, ovvero del controllo dell’accesso al mezzo condiviso

---
## Errori su bit
Gli errori sui bit sono dovuti a interferenze che possono cambiare la forma del segnale
La probabilità che avvenga un errore di tipo **burst** (a raffica) è più elevata rispetto a quella di un errore sul singolo bit, in quanto la durata dell’interferenza (detta anche rumore) normalmente è più lunga rispetto a quella di un solo bit

![[Pasted image 20250509165310.png|400]]

Il numero di bit coinvolti dipende dalla velocità di trasferimento dati e dalla durata del rumore

>[!example]
>$1\text{ kbps}$ con un rumore di $\frac{1}{100} \text{ sec}$ può influire su $10\text{ bit}$

### Tecniche di rilevazione degli errori
Per proteggere dei dati da errori vengono aggiunti dei bit *EDC* (**Error Detection and Corretion**)

Però la rilevazione degli errori non è attendibile al 100%, infatti è possibile che ci siano errori non rilevati. Per ridurre la possibilità di questo evento, le tecniche più sofisticare prevedono un’elevata **ridondanza**

![[Pasted image 20250509165646.png|450]]

#### Controllo di parità
Il bit aggiuntivo (di parità) viene selezionato in modo da rendere pari il numero totale di $1$ all’interno della codeword

Con un’unico bit di parità si può solo avere la certezza che si sia verificato almeno un errore in un bit, mentre tramite la **parità bidimensionale** è possibile individuare e correggere il bit alterato

![[Pasted image 20250509165921.png|420]]

---
## Protocolli di accesso multiplo
Esistono due tipi di collegamenti di rete:
- **collegamento punto-punto**, impiegato per
	- connessioni telefoniche
	- collegamenti punto-punto tra Ethernet e host
	- point-to-point protocol (PPP) del DLC
- **collegamento broadcast** (cavo o canale condiviso), impiegato per
	- Ethernet tradizionale
	- Wireless LAN 802.11

In una connessione a un canale broadcast condiviso, centinaia o anche migliaia di nodi possono comunicare direttamente su un canale broadcast e si genera una collisione quando i nodi ricevono due o più frame contemporaneamente

Con i protocolli di accesso multiplo l’obiettivo è quello di evitare caos e realizzare una condivisione
