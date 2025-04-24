---
Created: 2025-04-24
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
---
## Recap [[Forwarding dei datagrammi IP|forwarding datagrammi IP]]
Inoltrare significa collocare il datagramma sul giusto percorso (porta di uscita del router) che lo porterà a destinazione (o lo farà avanzare verso la prossima destinazione)

Quando un host ha un datagramma da inviare lo invia al router della rete locale. Quando un router riceve un datagramma da inoltrare accede alla tabella di routing per trovare il successivo hop a cui inviarlo.
L’inoltro richiede una riga della tabella per ogni blocco di rete

---
## Introduction

>[!question] Quale percorso deve seguire un pacchetto che viene instradato da un router sorgente a un router destinazione? Se sono disponibili più percorsi, quale si sceglie?

