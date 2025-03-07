---
Created: 2024-12-07
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Gestione della concorrenza]]"
Completed:
---
---
## Introduction
La condivisione di risorse può quindi essere implementata con 3 metodi diversi:
- istruzioni hardware (hanno il problema di attesa-attiva e starvation)
- sincronizzazione (semafori, starvation solvibile con coda forte)
- message passing (starvation solvibile con message passing forte)
Si può dimostrare che se è possibile implementare un’applicazione con uno qualsiasi dei 3 metodi, allora lo si può fare anche con gli altri due (un particolare meccanismo potrà rivelarsi più conveniente degli altri, in termini di facilità di sviluppo, di prestazioni, e di gestione)