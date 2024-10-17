---
Created: 2024-10-17
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduciamo $\textcolor{Peach}{\text{F}^\text{A}}$
Ricordiamo che il nostro problema è calcolare l’insieme di dipendenze $F^+$ che viene **soddisfatto da ogni istanza legale** di uno schema $R$ su cui è definito un insieme di dipendenze funzionali $F$.
Abbiamo concluso che banalmente $F\subseteq F^+$ in quanto una istanza è legale solo se soddisfa tutte le dipendenze in $F$

---
## Assiomi di Armstrong
Denotiamo con $F^A$ l’insieme di dipendenze funzionali definito nel modo seguente:
- se $f \in F$ allora $f \in F^A$
- se $Y \subseteq X \subseteq R$ allora $X \rightarrow Y \in F^A$ (**assioma della riflessività**, dipendenze funzionali banali)
- se $X\rightarrow Y \in F^A$ allora $XZ \rightarrow YZ \in F^A$, per ogni $Z \subseteq R$ (**assioma dell’aumento**)
- se $X\rightarrow Y \in F^A$ e $Y\rightarrow Z \in F^A$ allora $X\rightarrow Z \in F^A$ (**assioma della transitività**)

Dimostreremo che $\mathbf{F^+=F^A}$, cioè la chiusura di un insieme di dipendenze funzionali $F$ può essere ottenuta a partire da $F$ applicando ricorsivamente gli assiomi della riflessività, dell’aumento e della transitività, conosciuti come **assiomi di Armstrong**

