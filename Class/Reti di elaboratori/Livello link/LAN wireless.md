---
Created: 2025-05-17
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello link]]"
---
---
## Introduction
Esistono diversi tipi di reti wireless:
- LAN wireless → disponibili in campus universitari, uffici, bar, aree pubbliche
- reti cellulari → il numero di abbonati alla telefonia mobile supera quello della telefonia fissa
- bluetooth
- reti di sensori, RFID, smart ojects

Ecco alcuni degli standard esistenti
![[Pasted image 20250517105948.png]]

Quello più famoso è lo standard IEEE:

| Protocol | Release date | Freq.     | Rate (typical) | Rate (max) | Range (indoor) |
| -------- | ------------ | --------- | -------------- | ---------- | -------------- |
| Legacy   | 1997         | 2.4 GHz   | 1 Mbps         | 2 Mbps     | ?              |
| 802.11a  | 1999         | 5 GHz     | 25 Mbps        | 54 Mbps    | ~30 m          |
| 802.11b  | 1999         | 2.4 GHz   | 6.5 Mbps       | 11 Mbps    | ~30 m          |
| 802.11g  | 2003         | 2.4 GHz   | 25 Mbps        | 54 Mbps    | ~30 m          |
| 802.11n  | 2008         | 2.4/5 GHz | 200 Mbps       | 540 Mbps   | ~50 m          |

---
## Elementi
Una LAN wireless è composta da:
- *wireless host* → utilizzato per eseguire applicazioni e può essere stazionario o mobile (wireless non vuol dire necessariamente mobile)
- *base station* → tipicamente connesso ad una rete cablata; rasponsabile di mandare pacchetti tra la rete cablata e gli host wireless nella propria “area”
- *wireless link* → tipicamente usato per connettere gli host alla base station (usato anche come collegamento con il backbone); i protocolli ad accesso multiplo regolano l’accesso al link ed ha vari rates e range di trasmissione

![[Screenshot 2025-05-17 at 11.09.15.png]]

---
## Caratteristiche
Si hanno numerose differenze rispetto ad una classica rete calbata:
- mezzo trasmissivo → aria, segnale broadcast, mezzo condiviso dagli host della rete
- host wireless → non è fisicamente connesso alla rete e può muoversi liberamente
- connessione ad altre reti → mediante una stazione base detta Access Point (AP) che unisce l’ambiente wireless all’ambiente cablato

### Migrazione dall’ambiente cablato al wireless
Il funzionamento di una rete cablata o wireless dipende dai due sottolivelli inferiori dello stack protocollare (collegamento e fisico)

Per migrare dalla rete cablata a quella wireless è sufficiente cambiare le schede di rete e sostituire lo switch di collegamento con un AP (gli indirizzi MAC cambiano mentre gli IP rimangono gli stessi)

---
## Reti ad hoc
Le **reti ad hoc** (senza infrastruttura, senza un hotspot) sono composte da un insieme di host che si auto-organizzano per formare una rete e comunicano liberamente tra di loro (ogni host deve eseguire le funzionalità di rete quali network setup, routing, forwarding, etc.)

![[Pasted image 20250517112420.png|250]]

---
## Caratteristiche del link wireless
### Attenuazione del segnale
La forza dei segnali elettromagnetici diminuisce rapidamente all’aumentare della distanza dal trasmettitore in quanto il segnale si disperde in tutte le direzioni, fino ad arrivare al punto in cui la trasmissione non è più decodificabile

![[Pasted image 20250517112628.png|400]]

### Propagazione multi-path
Quando un’onda radio trova un ostacolo, tutta o una parte dell’onda è riflessa, con una perdita di potenza
Un segnale sorgente può arrivare, tramite riflessi successivi (su muri, terreno, oggetti), a raggiungere una stazione o un punto di accesso attraverso percorsi multipli

![[Pasted image 20250517124218.png|350]]

### Interferenze
Si possono avere due tipi di interferenze:
- dalla stessa sorgente → un destinatario può ricevere più segnali dal mittente desiderato a causa del multipath
- da altre sorgenti → altri trasmettitori stanno usando la stessa banda di frequenza per comunicare con altri destinatari

---
## Errori
Le caratteristiche dei link wireless causano errori. Il tasso di errore è misurato con il *Signal to Noise Radio* (**SNR**, o rapporto segnale-rumore) che misura il rapporto tra il segnale buono e il rumore esterno

Se l’SNR è alto, il segnale è più forte del rumore, quindi può essere convertito in dati reali, se invece è basso il segnale è stato danneggiato dal rumore e i dati non possono essere recuperati

### Controllo dell’accesso al mezzo condiviso
Per evitare collisioni (trasmissioni che si sovrappongono) è necessario controllare l’accesso al mezzo

>[!question] Perche non si può usare il CSMA/CD anche per le reti wireless?
>##### No collision detection
>Per rilevare una collisione un host deve poter trasmettere (il proprio frame) e ricevere (ascoltare il canale) conteportaneamente
>
>Poiché la potenza del segnale ricevuto è molto inferiore a quella del segnale trasmesso, sarebbe troppo costoso usare un adattatore in grado di rilevare le collisioni (i dispositivi wireless hanno un’energia limitata fornita dalla batteria che non gli consente di usare un tale dispositvo)
>
>##### Hidden terminal problem
>Un host potrebbe non accorgersi che un altro host sta trasmettendo e quindi non sarebbe in grado rilevare la collisione (ascoltando il canale)
>
>Attenuazione del segnale:
>![[Pasted image 20250517125050.png|350]]
>
>Ostacoli:
>![[Pasted image 20250517125116.png|250]]

---
## IEEE 802.11
La IEEE ha definito le specifiche per le LAN wireless, chiamate $802.11$, che coprono i livelli fisico e collegamento

Il Wi-Fi (wireless fidelity) è una LAN wireless dalla Wi-Fi Alliance, ovvero un’associazione ($300$ aziende) no profit che si occupa di promuovere la crescita delle LAN wireless

---
## Architettura
### BSS
La **BSS** (*Basic Service Set*) è una LAN wireless costituita da uno o più host wireless e da un access point

![[Pasted image 20250517125454.png]]

### ESS
La **ESS** (*Extended Service Set*) è una LAN wireless costituita da due o più BSS con infrastruttura

I BSS sono collegati tramite un sistema di distribuzione che è una rete cablata (Ethernet) o wireless. Quando i BSS sono collegati, le stazioni in visibilità comunicano direttamente, mentre le altre comunicano tramite l’AP

La ESS è un’architettura molto comune nelle reti Wi-Fi moderne, soprattutto in ambienti dove è necessario coprire aree estese con accesso continuo alla rete wireless

![[Pasted image 20250517125651.png|580]]

### Architettura generale
![[Pasted image 20250517125946.png|400]]

---
## Canali e associazione
Lo spettro $2.4 \text{ GHz}-2.485 \text{ GHz}$ è diviso in $11$ canali parzialmente sovrapposti
L’amministratore dell’AP sceglie una frequenza, ma sono possibili delle interferenze (stesso canale per AP vicini) e il numero massimo di frequenza utilizzabili da diversi AP per evitare interferenze è 