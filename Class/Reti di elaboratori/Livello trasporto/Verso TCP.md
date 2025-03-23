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

