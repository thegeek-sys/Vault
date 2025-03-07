---
Created: 2025-03-07
Class: "[[Algoritmi]]"
Related:
  - "[[Grafi]]"
Completed:
---
---
## Introduction
Dato un grafo connesso $G$ ed un intero $k$ vogliamo sapere se è possibile colorare i nodi del grafo in modo che i nodi adiacenti abbiano sempre colori distinti

>[!example] Esempio di grafo 3-colorabile
>![[Pasted image 20250307103527.png|550]]

---
## Teorema dei 4 colori

>[!info] Teorema dei 4 colori
>Un grafo **planare** richiede al più 4 colori per essere colorato

Il problema venne posto per la prima volta nel 1852 da uno studente che congetturò che 4 colori sono sempre sufficienti. Negli anni successivi molti matematici tentarono invano di dimostrare la congettura.
La prima dimostrazione fu proposta solo nel 1879, ma nel 1890 si scoprì che la dimostrazione conteneva un sottile errore. Si provò almeno che 5 colori sono sempre sufficienti a colorare una mappa (tramite un’induzione).
La dimostrazione che 4 colori sono sufficienti fu trovata solo nel 1977. Si basa sulla riduzione del numero infinito di mappe possibili a 1936 configurazioni, per le quali la validità del teorema viene verificata caso per caso con l’ausilio di un calcolatore
Nel 2000 infine è stata proposta una nuova dimostrazione del teorema che richiede l’utilizzo della teoria dei gruppi

In generale si può dire che un grafo può richiedere anche $\theta(n)$ colori, e inoltre non è noto alcun algoritmo polinomiale che, dato un grafo planare $G$, determini se $G$ è 3-colorabile
