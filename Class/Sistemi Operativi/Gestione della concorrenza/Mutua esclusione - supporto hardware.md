---
Created: 2024-11-28
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Gestione della concorrenza]]"
Completed:
---
---
## Introduction
In questa sezione vedremo dei modi funzionanti (a differenza della sezione precedente) per far rispettare la mutua esclusione

---
## Disabilitazione delle interruzioni

```c
while (true) {
	/* prima della sezione critica */;
	disabilita_interrupt() ;
	/* sezione critica */;
	riabilita_interrupt() ;
	/* rimanente */;
}
```

![[Pasted image 20241128233831.png|520]]

Disabilitando le interruzioni evito che il dispatcher interrompa il processo mentre si trova all’interno della sezione critica, però ci sono diversi problemi. 

Uno dei problemi più evidenti è che, se questa possibilità fosse concessa a tutti i processi utente, questi ne abusino riducendo così la multiprogrammazione.
Un ulteriore problema è che questo metodo funziona localmente sul singolo processore, quindi disabilitare le interruzioni su un singolo processore, non le disabilita sugli altri; quindi un altro processo potrebbe accedere alla sezione critica semplicemente perché viene messo in esecuzione su un altro processore, eseguendo una corsa critca

---
## Istruzioni macchina speciali
Per risolvere i due problemi sopracitati potrei utilizzare delle istruzioni macchina speciali come `compare_and_swap` e la `exchange` entrambe **atomiche** (l’hardware garantisce che un solo processo per volta possa eseguire una chiamata a tali funzioni/interruzioni anche se ci sono più processori)

#### `compare_and_swap`
Se il valore di `word` è uguale al valore di `testval` allora cambio il valore di `word` in `newval` e ritorno in ogni caso il precedente valore di `word`

```c
int compare_and_swap(int word, int testval, int newval) {
	int oldval;
	oldval = word;
	if (word == testval) word = newval;
	return oldval
}
```


#### `exchange`
La funzione `exchange` ha il compito di scambiare il contenuto di due argomenti (indirizzi di memoria)

```c
void exchange(int register, int memory) {
	int temp;
	temp = memory;
	register = temp;
}
```