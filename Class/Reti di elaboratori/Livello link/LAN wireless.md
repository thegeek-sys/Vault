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

