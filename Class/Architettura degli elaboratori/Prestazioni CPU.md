---
Created: 2024-03-05
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed: true
---
---

>[!info] Index
> - [[#Parametri|Parametri]]
>- [[#Formule|Formule]]
>- [[#Legge di Amdahl|Legge di Amdahl]]

---
## Parametri
Per misurare le prestazioni di una CPU utilizziamo i seguenti parametri:

| Componente delle prestazioni                        | Unità di misura                               |
| --------------------------------------------------- | --------------------------------------------- |
| Tempo di esecuzione della CPU per un dato programma | Secondi per programma                         |
| Numero di istruzioni                                | Istruzioni eseguite per singolo programma     |
| Cicli di clock per istruzione (CPI)                 | Numero medio di cicli di clock per istruzione |
| Durata del ciclo di clock                           | Secondi per ciclo di clock                    |

---
## Formule

$$
\text{Prestazioni}_{x} = \frac{1}{\text{Tempo di esecuzione}_{x}}
$$
$$
\begin{split}
& \text{Tempo di CPU relativo}\\
& \text{a un programma}
\end{split}\;=\;
\begin{split}
& \text{Cicli di clock}\\
& \text{della CPU relativi}\\
& \text{al programma}
\end{split}\;\times\;
\begin{split}
& \text{Periodo del clock}
\end{split}
$$
$$
\begin{split}
& \text{Tempo di CPU relativo}\\
& \text{a un programma}
\end{split}\;=\;
\frac{\text{Cicli di clock della CPU relativi al programma}}{\text{Frequenza del clock}}
$$
$$
\begin{split}
& \text{Cicli di clock della CPU}
\end{split}\;=\;
\begin{split}
& \text{Numero di istruzioni}\\
& \text{del programma}
\end{split}\;\times\;
\begin{split}
& \text{Periodo del clock}\\
& \text{clock per istruzione}
\end{split}
$$
$$
\text{Tempo di CPU} \;=\; \frac{\text{Numero di istruzioni}\;\times\;\text{CPI}}{\text{Frequenza del clock}}
$$

---
## Legge di Amdahl

Legge utilizzata per stimare il miglioramento delle prestazioni

$$
\text{Tempo di esecuzione dopo il miglioramento}=
$$
$$
=\frac{\text{Tempo di esecuzione influenzato dal miglioramento}}{\text{Miglioramento}}\,+\,
\begin{split}
&\text{Tempo di esecuzione non}\\
&\text{influenzato dal miglioramento}
\end{split}
$$

>[!tip] 
>Una delle linee guida nella progettazione dell' hardware è descritta da un corollario della legge di Amdahl:
>- **"Make the common case fast"** (rendi veloce il caso più frequente)

---
