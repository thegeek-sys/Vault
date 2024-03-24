---
Created: 2024-03-21
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
>[!info] Index
>- [[#Introduction|Introduction]]
>- [[#Ingredienti|Ingredienti]]

---
## Introduction
Una **funzione** è un frammento di codice che riceve degli argomenti e calcola un risultato (utile per rendere il codice riusabile e modulare)

![[Screenshot 2024-03-19 alle 13.20.59.png|500]]
Una funzione (o procedura) in assembly:
- ha un indirizzo di partenza
- riceve uno o più argomenti
- svolge un calcolo
- ritorna il risultato
- continua la sua esecuzione dall’istruzione seguente a quella che l’ha chiamata

---
## Ingredienti
Gli ingredienti principali per la creazione di funzioni in Assembly sono i **salti incondizionati**. In particolare delle istruzioni:
- `jal etichetta` → quest’istruzione, oltre che fare un jump all’etichetta (della funzione) indicata, salverà nel registro `$ra` l’indirizzo del Program Counter da cui è stato chiamato il jump
- `jr $ra` → questa istruzione viene eseguita al termine del corpo della funzione, così facendo infatti tornerà all’indirizzo del Program Counter successivo a quello da cui ho chiamato la funzione

Per convenzione vengono utilizzati i registri `$a0,$a1,$a2,$a3` per passare valori in input alla funzione, mentre `$v0,$v1` per restituire valori dalla funzioni

>[!info] Convenzioni
>- `$t0,$t1...` → possono cambiare tra una chiamata e l’altra (temporary)
>- `$s0,$s1...` → non cambiano tra una chiamata e l’altra (saved) 

---
## Chiamate nidificate
Per come è strutturato MIPS è dunque chiaro notare come la quantità di informazioni che posso passare in input e ricevere in output da una funzione è estremamente limitato (4 registri in input, 4 valori da 32 bit o 2 da 64, 2 in output, 2 valori da 32 bit o 1 da 64).

Mi potrebbe dunque essere utile effettuare delle chiamate nidificate all’interno della funzione. Ma quindi, come ricostruisco la memoria in ritorno?
In generale **conviene preservare il precedente contenuto dei registri usati dalla funzione e ripristinarlo**
- meno vincoli alla funzione chiamante
- nelle funzioni che chiamano altre funzioni, che perderebbero il contenuto almeno di `$ra`. Infatti se per accedere ad una funzione ho dovuto eseguire l’istruzione `jal`, per farne una nidificata lo dovrò eseguire di nuovo, facendomi perdere quindi il primo indirizzo del PC

Le informazioni da preservare hanno un ciclo di vita caratteristico, dovuto al nidificarsi delle chiamate delle funzioni:
- salvo lo stato prima di chiamata 1
	- salvo stat prima di chiamata 2
		- …
	- ripristino stato prima di chiamata 2
- ripristino stato prima di chiamata 1
Lo stack si riempie nelle chiamate annidate

Questo è il comportamento di una pila (**stack** o LIFO), in cui aggiungere un elemento (**push**) e togliere l’ultimo inserito (**pop**) viene realizzato con un vettore di cui si tiene l’indirizzo dell’ultimo elemento occupato nel registro `$sp` (Stack Pointer)