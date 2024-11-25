---
Created: 2024-11-25
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Gestione della concorrenza]]"
Completed:
---
---
## Processi multipli
Per i SO moderni, è essenziale supportare più processi in esecuzione in uno di questi tre modi:
- multiprogrammazione
- multiprocessamento (*multiprocessing*)
- computazione distribuita (cluster)

Il grosso problema ora da affrontare è la **concorrenza**, ovvero gestire il modo con cui questi processi interagiscono
### Multiprogrammazione
Se c’è un solo processore, i processi si alternano nel suo uso (*interleaving*)
![[Pasted image 20241125223829.png|480]]

### Multiprocessing
Se c’è più di un processore, i processi si alternano (*interleaving*) nell’uso di un processore, e possono sovrapporsi nell’uso dei vari processori (*overlapping*)
![[Pasted image 20241125224056.png|480]]

---
## Concorrenza
La concorrenza si manifesta nelle seguenti occasioni:
- applicazioni multiple → c’è condivisione del tempo di calcolo (a carico del SO, le altre no)
- applicazioni strutturate per essere parallele → perché generano altri processi o perché sono organizzate in thread
- struttura del sistema operativo → gli stessi SO operativi sono costituiti da svariati processi o thread in esecuzione parallela

---
## Terminologia
Per poter affrontare il problema della concorrenza al meglio è necessario imparare della terminologia:
- **Mutua esclusione** → questo è 