---
Created: 2025-04-06
Class: "[[Algoritmi]]"
Related: 
Completed:
---
---
## Introduction
Per illustrare il progetto e l’analisi di una algoritmo greedy consideriamo un problema piuttosto semplice chiamato **selezione di attività**

Abbiamo una lista di $n$ attività da eseguire:
- ciascuna attività è caratterizzata da una coppia con il suo tempo di inizio ed il suo tempo di fine
- due attività sono *compatibili* se non si sovrappongono

Vogliamo trovare un sottoinsieme di attività compatibili di massima cardinalità

>[!example]
>Istanza del problema con $n=8$
>![[Pasted image 20250406161602.png|450]]
>![[Pasted image 20250406161639.png|450]]
>
>Per le $8$ attività l’insieme da selezionare è $\{b,e,h\}$. In questo caso è facile convincersi che non ci sono altri insiemi di $3$ attività compatibili e che non c’è alcun insieme di $4$ attività compatibili. In generale possono esistere diverse soluzioni ottime

Volendo utilizzare il paradigma greedy dovremmo trovare una regola semplice da calcolare, che ci permetta di effettuare ogni volta la scelta giusta.

Per questo problema ci sono diverse potenziali regole di scelta

