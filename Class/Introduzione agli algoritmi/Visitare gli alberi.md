---
Created: 2024-05-03
Class: Introduzione agli algoritmi
Related:
  - "[[Class/Introduzione agli algoritmi/Strutture dati#Alberi|Alberi]]"
---
---
## Introduction
Un’operazione basilare sugli alberi è l’accesso a tutti i suoi nodi, uno dopo l’altro, al fine di poter effettuare una specifica operazione su ciascun nodo.
Tale operazione sulle liste si effettua con una semplice iterazione, ma sugli alberi la situazione è più complessa dato che la loro struttura è ben più articolata.
L’accesso progressivo a tutti i nodi di un albero si chiama **visita dell’albero**.


Negli alberi binari si ha la possibilità di visitare l’albero in tre differenti maniere:
- **in pre-order** → il nodo è visitato prima di proseguire la visita nei suoi sottoalberi
- **in order** → il nodo è visitato dopo la visita del sottoalbero sinistro e prima di quella del sottoalbero destro
- **in post-order** → il nodo è visitato dopo entrambe le visite dei sottoalberi

### Pre-order
