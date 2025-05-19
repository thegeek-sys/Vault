---
Created: 2025-05-19
Class: "[[Algoritmi]]"
Related:
---
---
![[Pasted image 20250519162637.png]]

>[!info] Dimostrazione
>Supponiamo che la soluzione $SOL$ trovata tramite questa tecnica non sia ottimale. Vuol dire che esiste una seconda soluzione ottima, sia $SOL^*$ la soluzione ottimale che differisce per minor numero di flaconi da $SOL$
>
>Dimostreremo ora l’esistenza di $SOL'$ che differisce per minor numero rispetto a $SOL^*$, il che è assurdo.
>
>Siano $F_{1},F_{2},\dots,F_{n}$ i flaconi scelti da $SOL$ e sia $F_{i}$ il primo flacone scelto da $SOL$ ma non da $SOL^*$
>
>Se però $F_{i}$ è stato scelto dall’algoritmo greedy vuol dire che mancavano delle pillole da sistemare e che $F_{i}$ era il più grande flacone in cui metterle e quindi in $SOL^*$, avendo meno flaconi, ce ne è almeno uno più grande di $F_{i}$. Ma se è presente un flacone più capiente sarebbe dovuto esser scelto dall’algoritmo greedy

