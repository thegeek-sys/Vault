---
Created: 2024-03-14
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---

>[!info] Index
>- [[#Set istruzioni|Set istruzioni]]
>- [[#Direttive principali per l’assemblatore|Direttive principali per l’assemblatore]]
>- [[#Istruzioni condizionali (if-else)|Istruzioni condizionali (if-else)]]
>	- [[#Istruzioni condizionali (if-else)#Esempio|Esempio]]
>- [[#Realizzare iterazioni|Realizzare iterazioni]]
>	- [[#Realizzare iterazioni#do-while|do-while]]
>	- [[#Realizzare iterazioni#while-do|while-do]]
>	- [[#Realizzare iterazioni#for loop|for loop]]
>	- [[#Realizzare iterazioni#switch-case|switch-case]]

---
## Set istruzioni
[[MIPS_Instruction_Set.pdf]]

>[!info]
>la maggior parte delle istruzioni in Assembly sono solamente delle pseudoistruzioni, infatti, come si vede nell’interfaccia di MARS, le istruzioni digitate vengono spesso trasformate in più istruzioni 

---
## Direttive principali per l’assemblatore
- `.data` → definizione dei dati statici (in memoria)
- `.text` → definizione del programma

- `.asciiz` → stringa terminata da zero
- `.byte` → sequenza di byte
- `.double` → sequenza di double
- `.float` → sequenza di float
- `.half` → sequenza di half words
- `.word` → sequenza di words
- `.globl sym` → dichiara il simbolo come globale e può essere referenziato da altri file

> [!info]
> La differenza tra un vettore di word e uno di halfword sta nel fatto che mentre nelle word per spostarmi tra un indice al successivo del vettore mi devo spostare di 4 byte, in un vettore di halfword mi basta spostarmi di due 2 byte alla volta (in in un vettore di byte di uno in uno).

---
## Istruzioni condizionali (if-else)
Esempio in C
```c
if (x > 0) {
	// codice da eseguire se il test è vero
} else {
	// codice da eseguire se il test è falso
}

// codice seguente
```

Esempio Assembly
```arm-asm
.text
# uso il registro $t0 per la var X

blez $t0,else            # salta se $t0 <= 0
	# codice da eseguire se il test è vero
	j endIf              # ho bisogno del jump altrimenti
						 # esegue anche l'else
else:
	# codice da eseguire se il test è falso
endIf:
	# codice seguente

```

### Esempio

```c
if (i == j) f = g + h; else f = g - h
```

```asm-arm
bne $s3,$s4,Else   # vai a Else se i ≠ j
add $s0,$s1,$s2    # f=g+h (saltata se i ≠ j)
j Esci             # vai a Esci
Else:
sub $s0,$s1,$s2    # f=g-h (saltata se i == j)
Esci:
# altro codice
```

---
## Realizzare iterazioni
### do-while
Esempio in C
```c
do {
	// codice da ripetere se x != 0
	// il corpo del ciclo DEVE aggiornare x
} while (x != 0)

// codice seguente
```

Esempio in Assembly
```asm-arm
.text
# uso il registro $t0 per indice x

do:
	# codice da ripetere
	
	bnez $t0,do      # test x != 0

# codice seguente
```

### while-do
Esempio in C
```c
while (x != 0) {
	// codice da ripetere se x != 0
	// il corpo del ciclo DEVE aggiornare x
}

// codice seguente
```

Esempio in Assembly
```asm-arm
.tex
# uso il registro $t0 per indice x

while:
	beqz $t0,endWhile    # esci se x = 0
	# codice da ripetere
	j while
endWhile:
	# codice seguente
```

### for loop
Esempio in C
```c
for (i=0; i<N; i++) {
	// codice da ripetere
}

// codice seguente
```

Esempio in Assembly
```asm-arm
.text
# uso il registro $t0 per indice i
# uso il registro $t0 per il limite N

xor $t0,$t0,$t0          # azzero i
li $t1,N                 # limite del ciclo
for:
	bge $t0,$t1,endFor
	# codice da ripetere
	addi $t0,$t0,1
	j for
endFor:
	# codice seguente
```

### switch-case
Esempio in C
```c
switch (A) {
	case 0:     // codice del caso 0
		break;
	case 1:     // codice del caso 1
		break;
	case N:     // codice del caso N
		break;
}

//  codice seguente
```

Esempio in Assembly
```arm-asm
.text
sll $t0,$t0,2       # A*4
lw $t1,dest($t0)    # carico indirizzo +$t0
jr $t1              # salto al registro

caso0:              # codice caso 0
	j endSwitch
caso1:              # codice caso 1
	j endSwitch
# altri casi
casoN:              # codice caso N
	j endSwitch
endSwitch:
	# codice seguente

.data
dest: .word caso0,caso1,……,casoN
```