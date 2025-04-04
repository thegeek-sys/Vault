---
Created: 2025-04-04
Class: "[[Reti di elaboratori]]"
Related: 
Completed:
---
---
## Introduction
Nel paradigma client/server la comunicazione a livello applicazione avviene tra due programmi applicativi in esecuzione chiamati processi: un client e un server
- un client è un programma in esecuzione che inizia la comunicazione inviando una richiesta
- un server è un altro programma applicativo che attende le richieste dai client

---
## API
Un linguaggio di programmazione prevede un insieme di istruzioni matematiche (un insieme di istruzioni per la manipolazione delle stringhe etc.)
Se si vuole sviluppare un programma capace di comunicare con un altro programma, è necessario un nuovi insieme di istruzioni per chiedere ai primi quattro livelli dello stack TCP/IP di aprire la connessione, inviare/ricevere dati e chiudere la connessione

Un insieme di istruzioni di questo tipo viene chiamato **API** (*Application Programming Interface*)

![[Pasted image 20250404115437.png|550]]

---
## Comunicazione tra processi
La comunicazione tra i processi avviene tramite il **socket**

Il **socket** appare come un terminale o un file ma non è un’entità fisica. E’ infatti una struttura dati creata ed utilizzata dal programma applicativo per comunicare tra un processo client e un processo server (equivale a comunicare tra due socket create nei due socket create nei due lati di comunicazione)

![[Pasted image 20250323170213.png]]

Un socket address è composto da indirizzo IP e numero di porta
![[Pasted image 20250323170357.png|500]]

---
## Indirizzamento dei processi
Affinché un processo su un host invii un messaggio a un processo su un altro host, il mittente deve identificare il processo destinatario