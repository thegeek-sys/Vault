---
Created: 2023-11-15
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
La **programmazione funzionale**, a differenza della programmazione strutturale, decompone il problema in un **insieme di funzioni** (queste sono dette *purely functional*).
Essendo *purely functional* non avrà side effects infatti qui non verranno usate strutture dati che vengono aggiornate via via che il programma esegue e l’output della funzione dovrà dipendere solamente dall’input (linguaggi Haskell, Lisp sono propriamente purely functional, Python non lo è).

---
## Iterator, Iterable, Iteration (Eager)
In Python esistono due metodi di iteazione:
- **eager** → iteratori (visti fin’ora)
- **lazy** → generatori