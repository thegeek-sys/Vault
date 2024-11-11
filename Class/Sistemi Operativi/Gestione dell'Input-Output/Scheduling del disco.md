---
Created: 2024-11-11
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## HDD vs SSD
Come abbiamo visto per gestire l’I/O il SO deve essere il più efficiente possibile.
Uno degli ambiti su cui i progettisti di sistemi operativi si sono dati più da fare è quello dei dispositivi di archiviazione di massa

Le tecniche che tratteremo riguardano solo e unicamente gli HDD (non gli SSD). Gli SSD hanno diversi problemi, che però non tratteremo

![[Pasted image 20241111144201.png]]

---
## Il disco
![[Pasted image 20241111144232.png|center]]

Una *traccia* è una corona circolare, avvicinandosi al centro diventa sempre più piccola
Un *settore* è una parte di cerchio delimitata da due raggi

Chiaramente i settori e le tracce non sono contigui infatti si ha l’*intertrack gap* che separa due tracce, mentre l’*intersector gap* separa due settori. Sono presenti in quanto altrimenti non sarebbe possibile leggere e scrivere i dati con grande accuratezza

---
## Come funziona un disco
I dati si trovano sulle tracce (corone concentriche) su un certo numero di settori. Quindi per leggere/scrivere occorre sapere su quale traccia si trovano i dati, e sulla traccia, su quale settore.

Per selezionare una traccia bisogna:
- spostare la testina, se il disco ha testine mobili
- selezionare una testina, se il disco a testine fisse
Per selezionare un settore su una traccia bisogna aspettare che il disco ruoti (a velocità costante)
Se i dati sono tanti, potrebbero essere su più settori o addirittura su più tracce (tipicamente un settore misura 512 bytes)

---
## Prestazioni del disco
La linea temporale di un disco può essere riassunta come segue
![[Pasted image 20241111204030.png]]