---
Created: 2024-03-17
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
>[!info] Index
>- [[#Introduction|Introduction]]
>- [[#American Standard Code for Information Interchange (ASCII)|American Standard Code for Information Interchange (ASCII)]]
>- [[#Endianess|Endianess]]
>	- [[#Endianess#Esempio|Esempio]]
>- [[#Accesso agli elementi|Accesso agli elementi]]
>	- [[#Accesso agli elementi#Esempio (con indice)|Esempio (con indice)]]
>	- [[#Accesso agli elementi#Esempio (con puntatore)|Esempio (con puntatore)]]

---
## Introduction
Per allocare una stringa in memoria, quello che facciamo altro non è che creare un vettore di byte, in cui, in ogni byte è memorizzata un carattere codificato secondo lo standard ASCII.
C’è però da notare che alla fine della stringa verrà salvato un carattere di fine stringa `\0` che nonostante rappresenti un carattere vuoto, è diverso dal carattere di spaziatura (`blank`)

Per esempio:
`label: .asciiz "sopra la panca"`
![[Screenshot 2024-03-17 alle 17.19.02.png]]

---
## American Standard Code for Information Interchange (ASCII)
![[Screenshot 2024-03-17 alle 17.22.54.png]]
https://theasciicode.com.ar

---
## Endianess
Il concetto di **endianess** (se un dato sistema è little endian o big endian) ha a che fare con:
- indirizzamento in memoria (che avviene per byte, ogni indirizzo specifica il singolo byte)
- come i byte di una word sono ordinati in memoria

Il processore MIPS permette l’ordinamento dei byte di una word in due modi:
- **Big-endian** (o network-order, usato da Java e dalle CPU SPARC Sun/Oracle)
	i byte della word sono memorizzati **dal most** significant byte **al least** significant byte
	![[Screenshot 2024-03-17 alle 17.30.01.png]]
	
- **Little-endian** (usato dalle CPU Intel, ad es. Windows, e da MARS)
	i byte della word sono memorizzati **dal least** significant byte **al most** significant byte
	![[Screenshot 2024-03-17 alle 17.31.27.png]]

### Esempio
`label: .asciiz "sopra la panca."`
![[Screenshot 2024-03-17 alle 17.33.49.png|500]]

`label: .word 0x6369616F  # "ciao" in hex`
![[Screenshot 2024-03-17 alle 17.52.22.png|150]]

---
## Accesso agli elementi
Per accedere ad un elemento di un vettore tramite indirizzo di memoria:
```arm-asm
# $t0 contiene l'indice dell'elemento (e.g. 2)
# $t1 contiene l'indirizzo del vettore (e.g. 0x10010040)
# in $t2 salvo l'indirizzo dell'elemento ($t1+offset)

sll $t2,$t0,2 # word -> 4 byte; shift di due bit -> 
			  # moltiplicazione per 4

add $t2,$t2,$t1 # essendo $t2 già indirizzo dell'indice
				# della word mi basta fare lw $s0,$t2 per
				# accedere t0-esimo elemento dell'array
```

Abbiamo quindi due metodi per accedere ad un elemento di un vettore:

| Scansione per indice                                                                                                                                                                                                                | Scansione per puntatore                                                                                                                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **PRO**<br>Comoda se si deve usare l’indice dell’elemento per controlli o altro<br><br>Incremento dell’indice non dipende dalla dimensione degli elementi<br><br>Comoda se il vettore è allocato staticamente (nella sezione .data) | **PRO**<br>Si lavora direttamente su indirizzi di memoria<br><br>Ci sono meno calcoli nel ciclo                                                                                                  |
| **CONTRO**<br>Bisogna convertire ogni volta l’indice nel corrispondente offset in byte                                                                                                                                              | **CONTRO**<br>Non si ha disposizione l’indice dell’elemento<br><br>L’incremento del puntatore dipende dalla dimensione degli elementi<br><br>Bisogna calcolare l’indirizzo successivo all’ultimo |

### Esempio (con indice)
Somma degli elementi di un vettore di word a posizione divisibile per tre
```arm-asm
.data
	vettore: .word 1,2,3,4,5,6,7,8,9
	N: .word 9
	Somma: .word 0
.text
  main:	li	$t0,0		       # i = 0
  	lw	$t1,N 		           # len(vettore)
  	li	$s0,0		           # somma=0
  loop:	 bge $t0,$t1,endLoop
  	sll  $t2,$t0,2		       # offset i*4
  	lw   $t2,vettore($t2)	   # vettore[i]
  	add  $s0,$s0,$t2		   # somma+=vettore[i]
  	addi $t0,$t0,3		       # i+=3
  	j loop
  endLoop:
  	sw $s0,Somma
```

### Esempio (con puntatore)

```arm-asm
.data
	vettore: .word 1,2,3,4,5,6,7,8,9
	N: .word 9
	Somma: .word 0
.text
  main:	la	$t0,vettore	  # indirizzo vettore
  	lw	$t1,N 			  # len(vettore)
  	sll	$t1,$t1,2		  # offset fine
  	add $t1,$t1,$t0		  # indirizzo fine
  	li	$s0,0
  loop:	bge $t0,$t1,endLoop
  	lw 	 $t2,($t0)		  # vettore[i]
  	add  $s0,$s0,$t2      # somma+=vettore[i]
  	addi $t0,$t0,12		  # i+=3*4
  	j loop
  endLoop:
  	sw $s0,Somma
```