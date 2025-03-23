---
Created: 2025-03-23
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello trasporto]]"
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Diagramma di comunicazione|Diagramma di comunicazione]]
- [[#Rappresentazione mediante FSM|Rappresentazione mediante FSM]]
- [[#Datagrammi UDP|Datagrammi UDP]]
	- [[#Datagrammi UDP#Struttura|Struttura]]
		- [[#Struttura#Checksum|Checksum]]
- [[#DNS usa UDP|DNS usa UDP]]
- [[#Ulteriori informazioni su UDP|Ulteriori informazioni su UDP]]
---
## Introduction
Il **protocollo UDP** (*User Datagram Protocol*) è un protocollo di trasporto inaffidabile e privo di connessione.
Questo fornisce servizi di:
- comunicazione tra processi utilizzando i socket
- multiplexing/demultiplexing dei pacchetti
- incapsulamento e decapsulamento (datagrammi indipendenti, non numerati)

Inoltre non fornisce alcun controllo di flusso, errori (eccetto *checksum*) e congestione

---
## Diagramma di comunicazione
Il mittente invia pacchetti uno dopo l’altro senza pensare al destinatario (può inviare dati a raffica perché non c’è controllo di flusso né di congestione)

![[Pasted image 20250323224341.png|center|600]]

Inoltre il mittente deve dividere i suoi messaggi in porzioni di dimensioni accettabili dal livello di trasporto, a cui consegnarli uno per uno. E’ importante ricordare che ogni pacchetto è **indipendente** dagli altri (la sequenza di arrivo può essere diversa da quella di spedizione, non c’è coordinazione tra livello di trasporto mittente e destinatario).

![[Pasted image 20250323224603.png|center|600]]

---
## Rappresentazione mediante FSM
Il comportamento di un protocollo di trasporto può essere rappresentato da un automa a stati finiti. L’automa rime in uno stato fin quando non avviene un evento che può modificare lo stato dell’automa (transizione di stato) e fargli compiere un’azione

![[Pasted image 20250323224801.png|600]]

Il protocollo UDP risulta essere molto semplice (rispetto al TCP)
![[Pasted image 20250323224901.png]]
Sarà dunque sufficiente uno stato per la nostra FSM
![[Pasted image 20250323224931.png|600]]

---
## Datagrammi UDP
I datagrammi UDP hanno delle dimensioni precise, un processo mittente non può inviare un flusso di dati e aspettarsi che UDP lo suddivida in datagrammi correlati
I processi inviati devono inviare richieste di dimensioni sufficientemente piccole per essere inserite ciascuna in un singolo datagramma utente

Solo i processi che usano messaggi di dimensione inferiore a $65507\text{ byte}$ ($65535-8\text{ byte}$ di intestazione UDP e $20 \text{ byte}$ di intestazione IP) possono utilizzare il protocollo UDP

### Struttura
Un messaggio UDP è composto da $4$ campi da $2\text{ byte}$ ciascuno per l’intestazione più il messaggio
![[Pasted image 20250323225422.png|400]]

#### Checksum
Il compito del checksum è quello di rilevare gli “errori” (bit alterati) nel datagrammi tramesso

| Mittente                                                                              | Ricevente                                                                                        |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1. Il messaggio viene diviso in “parole” da $16 \text{ bit}$                          | 1. Il messaggio (che comprende il checksum viene ricevuto)                                       |
| 2. Il valore del checksum viene inizialmente impostato a zero                         | 2. Il messaggio viene diviso in parole da $16 \text{ bit}$                                       |
| 3. Tutte le parole del messaggio vengono sommato usando l’addizione complemento a uno | 3. Tutte le parole, incluso il checksum, vengono sommate usando l’addizione complemento a uno    |
| 4. Viene fatto il complemento a uno della somma e il risultato è il checksum          | 4. Viene fatto il complemento a uno della somma e il risultato diventa il nuovo checksum         |
| 5. Il checksum viene inviato assieme ai dati                                          | 5. Se il valore del checksum è $0$ allora il messaggio viene accettato altrimenti viene scartato |
>[!warning]
>Quando si sommano i numeri, un riporto dal bit più significativo deve essere sommato al risultato

>[!example] Esempio di checksum
>Calcolare  il checksum della seguente stringa di $32\text{ bit}$
>![[Pasted image 20250323230702.png]]
>
>![[Pasted image 20250323230825.png]]

---
## DNS usa UDP
Quando vuole effettuare una query, DNS costruisce un messaggio di query e lo passa a UDP
L’entità UDP aggiunge i campi di intestazione al messaggio e trasferisce il segmento risultate al livello di rete etc.
L’applicazione DNS aspetta quindi una risposta. Se non ne riceve, tenta di inviarla a un altro server dei nomi.

In questo caso la semplicità della richiesta/risposta (molto breve) motiva l’utilizzo di UDP, che risulta più veloce poiché non viene stabilita nessuna connessione, non si ha nessuno stato di connessione e le intestazioni di pacchetto sono più corte rispetto al TCP

UDP in generale viene utilizzato anche perché consente un controllo più sottile a livello di applicazione su quali dati sono inviati e quando

---
## Ulteriori informazioni su UDP
Il protocollo UDP viene inoltre spesso utilizzato nelle applicazioni multimediali che infatti tollerano perite di pacchetti (anche se limitate)

