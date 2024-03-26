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
>- [[#Chiamate nidificate]]
>	- [[#Esempio chiamate funzioni nidificate]]
>- [[#Uso dello stack]]
>	- [[#In una funzione]]
>- [[#$fp e $sp]]

---
## Introduction
Una **funzione** è un frammento di codice che riceve degli argomenti e calcola un risultato (utile per rendere il codice riusabile e modulare)

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

### Esempio chiamate funzioni nidificate
- `main` chiama `foo` che chiama `bar`
- `foo` ha bisogno di 3 registri $s0, $s1, $s2
- `bar` ha bisogno di 2 registri $s0, $s1
- return address? (se ho jal nidificato lo perdo)

```arm-asm
main:
	...
	jal foo # $ra=PC+1

# sapendo che foo sporcherà tre registri vuol dire che
# dovrò salvare nello stack questi tre valori
# spostando il relativo stack pointer
foo:
	...
	jal bar # poiché questa istruzione cambierà $ra dovrò
			# salvarmi il precedente stato di $ra dentro
			# lo stack

bar:
	...
	jr $ra  # tornando indietro mi dovrò ricordare di 
			# svuotare lo stack e rimettere i valori 
			# precedenti (come $ra, $s0, $s1, $s2)
```

---
## Uso dello stack
![[Screenshot 2024-03-25 alle 21.16.02.png|500]]
Questo è il comportamento di una pila (**stack** o LIFO), in cui aggiungere un elemento (**push**) e togliere l’ultimo inserito (**pop**) viene realizzato con un vettore di cui si tiene l’indirizzo dell’ultimo elemento occupato nel registro `$sp` (Stack Pointer)
Lo stack si trova nella parte «alta» della memoria e cresce verso il basso. Supponiamo di voler salvare e ripristinare il registro `$ra`

Come salvare un elemento (push):
- si decrementa lo `$sp` della dimensione dell’elemento (in genere una word)
	`subi $sp,$sp,4`
- si memorizza l’elemento nella posizione 0(`$sp`)
	`sw $ra,0($sp)`

Come recuperare un elemento (pop):
- si legge l’elemento dalla posizione 0(`$sp`)
	`lw $ra,0($sp)`
- si incrementa lo `$sp` della quantità allocata in precedenza
	`addi $sp,$sp,4`

### In una funzione
All’inizio della funzione:
- allocare su stack abbastanza word da contenere i registri da preservare
- salvare su stack i registri, ad offset multipli di 4 rispetto a $sp
All’uscita della funzione:
- ripristinare da stack i registri salvati, agli stessi offset usati precedentemente
- disallocare da stack lo stesso spazio allocato in precedenza
- tornare alla funzione chiamante

```arm-asm
funzione:
	addi $sp,$sp,-12
	sw $ra,8($sp)
	sw $a0,4($sp)
	sw $a1,0($sp)
	
	# corpo della funzione
	# posso chiamare jal
	# tranquillamente perché
	# salvo $ra
	
	lw $a1, 0($sp)
	lw $a0, 4($sp)
	lw $ra, 8($sp)
	addi $sp, $sp, 12
	jr $ra
```

> [!info]
> Conviene allocare tutto lo spazio assieme per avere offset che restano costanti durante tutta l’esecuzione della funzione

## $fp e $sp
La differenza tra **stack pointer** e **frame pointer** sta nel fatto che mentre `$sp` si utilizza per puntare alla fine del record di attivazione (varia allocando dati dinamicamente) `$fp` si usa per puntare all’inizio del record di attivazione (resta fisso durante l’esecuzione della funzione, non molto usato)

![[Screenshot 2024-03-25 alle 21.14.58.png|530]]

