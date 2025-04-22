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

---
## Indirizzamento IPv4
Un **indirizzo IP** è formato da $32\text{ bit}$ ($4\text{ byte}$) in notazione decimale puntata (ciascun byte dell’indirizzo viene indicato in forma decimale) e ogni interfaccia di host e router di Internet ha un indirizzo IP globalmente univoco a $32\text{ bit}$

![[Pasted image 20250420111723.png|400]]

Una **interfaccia** è il confine tra host e collegamento fisico. I router devono necessariamente essere connessi ad almeno due collegamenti, un host invece, generalmente, ha un’interfaccia e a ciascuna di loro è associato un indirizzo IP

### Spazio degli indirizzi
Il numero totale degli indirizzi è $2^{32}$ ovvero più di $4$ miliardi. Per identificarli si usano diverse notazioni:
- binaria
- decimale puntata
- esadecimale (usata nella programmazione di rete)

![[Pasted image 20250420111857.png|450]]

---
## Gerarchia dell’indirizzo
Ogni indirizzo è composto da un prefisso e da un suffisso. Il prefisso può avere lunghezza:
- **fissa** → indirizzamento con classi
- **variabile** → indirizzamento senza classi

### Indirizzamento con classi
Questo viene usato sia per reti piccole che grandi e si hanno tre possibili lunghezze del prefisso: 8, 16 e 24

| Classe | Prefissi          | Primo byte       |
| ------ | ----------------- | ---------------- |
| A      | $n=8\text{ bit}$  | da $0$ a $127$   |
| B      | $n=16\text{ bit}$ | da $128$ a $191$ |
| C      | $n=24\text{ bit}$ | da $192$ a $223$ |
| D      | non applicabile   | da $224$ a $239$ |
| E      | non applicabile   | da $240$ a $255$ |
![[Pasted image 20250420112605.png|450]]

#### Pros e cons
**Pro**
Una volta individuato un indirizzo si può facilmente risalire alla classe e la lunghezza del prefisso

**Contro**
Con il passare del tempo ci si è resi conto che gli indirizzi andavano esaurendosi
- la classe A può essere assegnata solo a $128$ organizzazioni al mondo, ognuna con $16.777.216$ nodi. Così facendo la maggior parte degli indirizzi andava sprecata e solo poche organizzazione potevano usufruire  di indirizzi di classe A
- con la classe B si hanno gli stessi problemi della classe A
- con la classe C invece si hanno pochi indirizzi ($256$) per la rete

### Indirizzamento senza classi
Venne introdotto l’indirizzamento senza classi per la necessità di avere maggiore flessibilità nell’assegnamento degli indirizzi. Infatti, in questo caso, vengono utilizzati blocchi di lunghezza variabile che non appartengono a nessuna classe.

Così facendo però un indirizzo non è in grado di definire da solo la rete (o blocco) a cui appartiene per questo motivo la lunghezza del prefisso (da $0$ a $32\text{ bit}$) viene aggiunta all’indirizzo separata da uno slash

#### Notazione CIDR
Il **CIDR** (*Classless InterDomain Routing*) è la strategia adottata per l’assegnazione degli indirizzi. L’indirizzo IP in questo caso viene diviso in due parti e mantiene la forma decimale puntata $a.b.c.d/n$ dove $n$ indica il numero di bit nella prima parte dell’indirizzo

![[Pasted image 20250422173314.png|center|400]]

>[!example]
>![[Pasted image 20250422173352.png|400]]

#### Estrazione delle informazioni
Se $n$ è la lunghezza del prefisso:
1. il numero di indirizzi nel blocco è dato da $N=2^{32-n}$
2. per trovare il primo indirizzo si impostano a $0$ tutti i bit del suffisso ($32-n$)
3. per trovare l’ultimo indirizzo si impostano a $1$ tutti i bit del suffisso ($32-n$)

![[Pasted image 20250422173630.png]]

---
## Maschera e indirizzo di rete
La maschera dell’indirizzo è un numero composto da 32 bit in cui i primi $n$ bit a sinistra sono impostati a 1 e il resto ($32-n$) a 0 e permette di ottenere l’**indirizzo di rete che è usato nell’instradamento dei datagrammi verso la destinazione**

![[Pasted image 20250422174214.png]]

