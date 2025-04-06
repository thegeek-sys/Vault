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
## $\verb|while|$ loop
La condizione viene valutata prima di entrare nel loop e se `condizione==true` viene eseguito il body e poi viene rivalutata la condizione. Per fare in modo di non entrare in un loop infinito è necessario che un evento esterno o intero cambi la condizione. Infatti solo quando `condizione==false` il loop termina

![[Pasted image 20250406172356.png|center]]

```c
while (expression) {
	basic block
}
```
- `expression` → la condizione da testare
- `basi block` → racchiuso tra `{}` se contiene più di un’istruzione
