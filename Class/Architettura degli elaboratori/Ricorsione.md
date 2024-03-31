---
Created: 2024-03-31
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
>[!info] Index
>- [Introduction](#Introduction)
>- [Fattoriale ricorsivo](#Fattoriale%20ricorsivo)

---
## Introduction
La ricorsione si può utilizzare se esiste una soluzione conosciuta per lo stesso problema di “piccole” dimensioni (da questa ricaviamo il caso base della funzione) e se esiste un modo di ottenere la soluzione di un problema di dimensione maggiore a partire dalla soluzione dello stesso problema di dimensione minore; da questa seconda sefinizione ricaviamo il caso ricorsivo, che è formato da 3 parti:
- riduzione del problema in problemi “più piccoli”
- chiamata ricorsiva della funzione per risolvere i casi “più piccoli”
- elaborazione delle soluzioni “piccole” per ottenere la **soluzione del problema originale**

---
## Fattoriale ricorsivo

 ```arm-asm
.data
	N: .word 10

.text
main:
	lw $a0,N
	jal factorial

factorial:
	blez $a0,BaseCase
	
	RecursiveStep:
		subi $sp,$sp,8    # decrement the stack pointer
		sw $ra,0($sp)     # store the return address for later
		sw $a0,4($sp)     # store the value of n
		
		subi $a0,$a0,1    # decrement by 1
		jal factorial     # do the recursive call
		
		lw $a0,4($sp)     # restore n
		lw $ra,0($sp)     # restore the return address
		addi $sp,$sp,8    # restore the stack pointer
		
		mul $v0,$v0,$a0   # compute n * factorial(n-1)
		jr $ra            # return
		
	BaseCase:
		addi $v0,$zero,1  # 0! = 1! = 1
		jr $ra
```

> [!hint]
> se la ricorsione è multipla, un contatore non è sufficiente per ricostruire la struttura delle chiamate. Può essere necessario usare uno stack per simulare la gestione corretta delle chiamate

