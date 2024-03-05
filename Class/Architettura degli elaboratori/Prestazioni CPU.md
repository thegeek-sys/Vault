---
Created: 2024-03-05
Related: 
Completed:
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
\begin{equation}
\begin{split}
\text{Tempo di CPU relativo a un programma} \\
& \text{a un programma}
\end{split}
\end{equation}
$$




![[Pasted image 20240305150036.png|800]]

---
## Legge di Amdahl

Legge utilizzata per stimare il miglioramento delle prestazioni

![[Pasted image 20240305125326.png|700]]

>[!tip] 
>Una delle linee guida nella progettazione dell' hardware è descritta da un corollario della legge di Amdahl:
>- **"Make the common case fast"** (rendi veloce il caso più frequente)

---
