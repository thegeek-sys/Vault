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

