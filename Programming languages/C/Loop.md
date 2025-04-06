---
Created: 2025-04-06
Class: 
Programming language: 
Related:
---
---
## Introduction
Esistono tre tipi di costrutti iterativi in C:
- `while` statement → più flessibile, nessun tipo di restizione
- `for` statement → ciclo ripetuto un numero intero di volte
- `do-while` statement → esegue il body almeno una volta

---
## Loop Control Variable
La **Loop Control Variable** (*LCV*) è la variabile che il cui valore controlla la ripetizione del loop. Per una corretta esecuzione del loop la LCV deve:
- essere dichiarata e inizializzata prima del loop
- testata nella condizione del loop
- aggiornata nel body del loop in modo che prima o poi la condizione diventi falsa (altrimenti loop infinito)

---
## $\verb|while|$ loop
La condizione viene valutata prima di entrare nel loop e se `condizione==true` viene eseguito il body e poi viene rivalutata la condizione. Per fare in modo di non entrare in un loop infinito è necessario che un evento esterno o intero cambi la condizione. Infatti solo quando `condizione==false` il loop termina

![[Pasted image 20250406172356.png|center]]

```c
while (expression) {
	basic block
}
```
- `expression` → la condizione da testare
- `basic block` → racchiuso tra `{}` se contiene più di un’istruzione

---
## $\verb|for|$ loop
Nel `for` loop il numero di iterazioni è noto a priori (ma tale comportamento può essere sovvertito). Nella definizione dello statement del `for` si ha inizializzazione, test della condizione e incremento/decremento dell’LCV

```c
for (inizialization; test; increment) {
	basic block
}
```

---
## $\verb|do-while|$ loop
Simile al `while`, ma la condizione è testata solo alla fine, cosicché il body è eseguito almeno una volta

```c
do {
	basic block
} while (condition);
```

---
## $\verb|break|$ statement
Lo statement `break` viene usato per uscire da un ciclo e dallo `switch`. Quando viene eseguito il `break` il ciclo termina indipendentemente dal valore della condizione

![[Pasted image 20250406173841.png|150]]

---
## $\verb|continue|$ statement
Lo statement `continue` passa all’iterazione successiva

![[Pasted image 20250406174011.png|150]]

