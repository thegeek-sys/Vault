---
Created: 2024-03-18
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
>[!info] Index
>- [[#Introduction|Introduction]]
>	- [[#Introduction#Esempio|Esempio]]
>- [[#Matrice 3D|Matrice 3D]]

---
## Introduction
Una matrice $\text{M x N}$ altro non è che una successione di $\text{M}$ vettori, ciascuno di $\text{N}$ elementi:
- il numero di elementi totali è: $\text{M x N}$
- la dimensione totale in byte è $\text{M x N x dimensione-elemento}$
- la si definisce staticamente come un vettore contenente $\text{M x N}$ elementi uguali

### Esempio
`Matrice: .word 0:91 # spazio di una matrice di 7x13 word`
![[Screenshot 2024-03-18 alle 17.40.20.png|500]]

Ad esempio per accedere all’elemento `e` che si trova alle coordinate $x=9$, $y=2$, vuol dire che si trova ad una distanza di 2 righe e 9 elementi dall’inizio ovvero ad un offset di $2\cdot 13+9=35 \text{ word}$ cioè $35\cdot4=140 \text{ byte}$

---
## Matrice 3D
![[Screenshot 2024-03-15 alle 13.30.20.png|500]]
Una matrice 3D di dimensioni $\text{M x N x P}$ è una successione di $\text{P}$ matrici grandi $\text{M x N}$.
Il che vuol dire che l’elemento a coordinate $x,y,z$ è preceduto da:
- **z strati** → matrici $\text{M x N}$ formate da $M\cdot N$ elementi
- **y righe** → di $\text{M}$ elementi sullo stesso strato
- **x elementi** → sulla stessa riga e strato
Quindi l’elemento si trova $z * (M * N) + y * N + x$ elementi dall’inizio della matrice 3D
La sua posizione in memoria è $\text{indirizzo-matrice}+(z * (M * N) + y * N + x)*\text{dim-el}$
