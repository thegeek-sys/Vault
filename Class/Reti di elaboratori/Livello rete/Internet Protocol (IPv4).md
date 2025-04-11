---
Created: 2025-04-11
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Formato dei datagrammi|Formato dei datagrammi]]
- [[#Frammentazione|Frammentazione]]
	- [[#Frammentazione#Frammentazione dei datagrammi IP|Frammentazione dei datagrammi IP]]
	- [[#Frammentazione#Identificatore, flag e offset di frammentazione nel dettaglio|Identificatore, flag e offset di frammentazione nel dettaglio]]
	- [[#Frammentazione#Riassemblaggio a destinazione|Riassemblaggio a destinazione]]
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
Differenti tipi di link hanno differenti MTU dunque i datagrammi IP grandi vengono suddivisi (“frammentati”) in datagrammi IP più piccoli per poi essere riassemblati solo una volta raggiunta la destinazione (prima di raggiungere il livello di trasporto). Vengono dunque usati i bit dell’intestazione IP per identificare e ordinare i frammenti

![[Pasted image 20250411115723.png|350]]

### Identificatore, flag e offset di frammentazione nel dettaglio
Quando un host di destinazione riceve una serie di datagrammi dalla stessa origine deve:
- individuare i frammenti
- determinare quando ha ricevuto l’ultimo
- stabilire come debbano essere riassemblati

Per fare ciò si usano i campi **identificatore, flag e offset di frammentazione**
- **Identificazione** (16 bit) → identificativo associato a ciascun datagramma al momento della creazione (unico per tutti i frammenti)
- **Flag** (3 bit) → il primo è riservato, il secondo (*do not fragment*) indica se frammentare o no (1 → non frammentare, 0 si può frammentare) e il terzo (*more fragments*) indica se sono presenti altri frammenti dopo (1 → frammenti intermedi, 0 ultimo frammento)
- **Offset** (13 bit) → specifica l’ordine del frammento all’interno del datagramma originario

>[!example] Offset
>Payload di un datagramma con un dimensione di 4000 byte suddiviso in tre frammenti
>![[Pasted image 20250411115842.png]]
>
>>[!warning] L’offset dei dati nel datagramma originale è misurato in unità di 8 byte

>[!example] Esempio di frammentazione
>![[Pasted image 20250411115951.png]]
>
>In totale vengono trasferiti $40\text{ byte}$ in più (ci sono due header aggiuntivi)
>
>>[!example] Frammentazione di un frammento
>>![[Pasted image 20250411120101.png]]

### Riassemblaggio a destinazione
Il primo frammento ha un valore del campo offset pari a $0$. L’offset del secondo frammento si ottiene dividendo per $8$ la lunghezza del primo frammento (esclusa l’intestazione). Il valore del terzo frammento si ottiene dividendo per $8$ la somma della lunghezza del primo e del secondo frammento (escluse le
intestazioni). Si continua così finché non si raggiunge l’ultimo frammento che ha il bit $M$ (more fragments) impostato a $0$

