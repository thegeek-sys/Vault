---
Created: 2025-04-11
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
Completed:
---
---
## Introduction
L’**Internet Protocol** è responsabile della suddivisione, dell’inoltro e della consegna dei datagrammi a livello di rete (host to host).

Questo è inaffidabile, senza connessione e basato su datagrammi, offrendo un servizio di consegna *best effort*

---
## Formato dei datagrammi
![[Pasted image 20250411113914.png]]

- **Numero di versione** → consente al router la corretta interpretazione del datagramma (4 → IPv4, 6 → IPv6)
- **Lunghezza dell’intestazione** → poiché un datagramma IP può contenere un numero variabile di opzioni (incluse nell’intestazione), questi bit indicano dove inizia il campo dati (intestazione senza opzione è di $20 \text{ byte}$)
- **Tipo di servizio** → serve per distinguere diversi datagrammi con requisiti di qualità del servizio diverse (*realtime*)
- **Lunghezza del datagramma** → rappresenta la lunghezza totale del datagramma IP inclusa l’intestazione (in byte). In genere non superiore ai $1500\text{ byte}$ e serve per capire se il pacchetto è arrivato completamente
- **Identificatore, flag e offset di frammentazione** → questi tre campi servono per gestire la frammentazione dei pacchetti (IPv6 non prevede frammentazione)
- **Tempo di vita** → o time to live (TTL) è incluso per assicurare che i datagrammi non restino in circolazione per sempre nella rete (in caso per esempio di instradamento ciclico). Il campo viene decrementato a ogni hop e il datagramma viene eliminato in caso il suo $\text{TTL} = 0$
- **Protocollo** → indica il protocollo a livello di trasporto al quale va passato il datagramma. Questo campo è utilizzato solo quando il datagramma raggiunge la destinazione finale.
	- 6 → TCP
	- 17 → UDP
	- 1 → ICMP
	- 2 → IGMP
	- 89 → OSPF
- **Checksum dell’intestazione** → consente ai router di rilevare errori sui datagrammi ricevuti. Questa viene calcolata solo sull’intestazione e ricalcolata nei router intermedi (TTL e frammentazione); invece la checksum UDP/TCP è calcolata sull’intero segmento
- **Indirizzi IP di origine e destinazione** → inseriti dall’host che crea il datagramma (dopo aver effettuato una ricerca DNS)
- **Opzioni**: campi che consentono di estendere l’intestazione IP (usate per test o debug della rete)
- Dati: contiene il segmento di trasporto da consegnare alla destinazione e può trasportare altri tipi di dati, quali i messaggi ICMP, IGMP, ecc.

---
## Frammentazione
Un datagramma IP può dover viaggiare attraverso varie reti, ognuna con caratteristiche diverse. Ogni router estrae il datagramma dal frame, lo elabora e lo incapsula in un nuovo frame.

La **Maximum Transfer Unit** (*MTU*) è la massima quantità di dati che un frame a livello di collegamento può trasportare e varia in base alla tecnologia

![[Pasted image 20250411114824.png]]

### Frammentazione dei datagrammi IP
