---
Created: 2025-03-23
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello trasporto]]"
Completed:
---
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

| Mittente                                                                                                    | Ricevente                                                                                        |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1. Il messaggio viene diviso in “parole” da $16 \text{ bit}$                                                | 1. Il messaggio (che comprende il checksum viene ricevuto)                                       |
| 2. Il valore del checksum viene inizialmente impostato a zero                                               | 2. Il messaggio viene diviso in parole da $16 \text{ bit}$                                       |
| 3. Tutte le parole del messaggio, incluso il checksum, vengono sommato usando l’addizione complemento a uno | 3. Tutte le parole vengono sommate usando l’addizione complemento a uno                          |
| 4. Viene fatto il complemento a uno della somma e il risultato è il checksum                                | 4. Viene fatto il complemento a uno della somma e il risultato diventa il nuovo checksum         |
| 5. Il checksum viene inviato assieme ai dati                                                                | 5. Se il valore del checksum è $0$ allora il messaggio viene accettato altrimenti viene scartato |
>[!example] Esempio di checksum

