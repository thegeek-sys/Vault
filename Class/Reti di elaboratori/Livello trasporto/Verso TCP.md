---
Created: 2025-03-23
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello trasporto]]"
Completed:
---
---
## Diagramma di flusso a confronto
![[Pasted image 20250323231944.png]]

---
## Servizi del TCP
I servizi offerti dal TCP sono:
- Comunicazione tra processi (indirizzamento tramite numero di porta)
- Incapsulamento/decapsulamento
- Multiplexing/demultiplexing
- **Trasporto orientato alla connessione**
- **Controllo di flusso**
- **Controllo degli errori**
- **Controllo della congestione**

### Demultiplexing orientato alla connessione
La socket TCP è identificata da 4 parametri:
- indirizzo IP di origine
- numero di porta di origine
- indirizzo IP di destinazione
- numero di porta di destinazione

L’host ricevente usa i quattro parametri per inviare il segmento alla socket appropriata
Un host server può supportare più socket TCP contemporanee (ogni socket è identificata dai suoi 4 parametri)

I server web hanno socket differenti per ogni connessione client (con HTTP non-persistente si avrà una connessione differenza anche per ogni richiesta dallo stesso client)

![[Pasted image 20250323232811.png]]

>[!warning] Porta ≠ socket
>Sulla stessa porta possono essere attive più socket (nell’immagine sopra si hanno 3 socket sulla porta 80)

