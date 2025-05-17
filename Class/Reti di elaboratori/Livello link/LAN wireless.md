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
