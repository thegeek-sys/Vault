---
Created: 2024-03-17
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## Introduction
Per allocare una stringa in memoria, quello che facciamo altro non è che creare un vettore di byte, in cui, in ogni byte è memorizzata un carattere codificato secondo lo standard ASCII.
C’è però da notare che alla fine della stringa verrà salvato un carattere di fine stringa `\0` che nonostante rappresenti un carattere vuoto, è diverso dal carattere di spaziatura (`blank`)

Per esempio:
`label: .asciiz "sopra la panca"`
![[Screenshot 2024-03-17 alle 17.19.02.png|650]]

