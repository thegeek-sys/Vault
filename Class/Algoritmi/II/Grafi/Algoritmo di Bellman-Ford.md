---
Created: 2025-03-28
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Introduction

>[!question] Problema
>Dato un grafo diretto e pesato $G$ in cui i pesi degli archi possono essere anche negativi e fissato un suo nodo $s$, vogliamo determinare il costo minimo dei cammini che conducono da $s$ a tutti gli altri nodi del grafo. Se non esiste un cammino verso un determinato nodo il costo sarà infinito

In questo primo problema però non è considerata la possibilità di poter avere un **ciclo negativo**, ovvero un ciclo diretto in un grafo in cui la somma dei pesi degli archi che lo compongono è negativa

>[!example]
>![[Pasted image 20250328011833.png|350]]
>Il ciclo evidenziato in figura è negativo di costo $5-3-6+4-2=\textcolor{red}{-2}$

Infatti se in un cammino tra i nodi $s$ e $t$ è presente un nodo che appartiene ad un ciclo negativo allora non esiste un cammino minimo tra $s$ e $t$ (è $-\infty$)
