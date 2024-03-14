---
Created: 2024-03-14
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## Set istruzioni
[[MIPS_Instruction_Set.pdf]]

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

### for loopù
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

for:
	bge $t0,
```