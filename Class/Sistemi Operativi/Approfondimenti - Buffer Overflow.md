---
Created: 2024-12-16
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Introduction
L’area di memoria di un processo caricato in memoria è diviso nelle sezioni seguenti
![[Pasted image 20241216185400.png|140]]

### Stack
Lo stack nello specifico è costituito da **frames**. In ciascun frame sono contenuti i parametri passi alla funzioni, variabili locali, indirizzo di ritorno e instruction pointer
Sono inoltre presenti due puntatori:
- stack pointer → punta alla cima dello stack (indirizzo più basso)
- frame pointer → punta alla base del frame corrente

### Chiamata di funzione
Quando all’interno di una chiamata di funzione sono presenti dei parametri, questi sono aggiunti allo stack. Inoltre quando avviene una chiamata vengono memorizzati sullo stack anche:
- indirizzo di ritorno
- puntatore allo stack frame
- spazio ulteriore per le variabili locali della funzione chiamata

Nel caso in cui vengono chiamate due funzioni $G$ e $F$ si ha
![[Pasted image 20241216185902.png|400]]

---
## Il problema
Cosa succede adesso?

```c
void foo(char *s) {
	char buf[10];
	strcpy(buf, s);
	printf("buf is %s\n", s);
}

foo("stringatroppolungaperbuf");
```

In questo caso stiamo inserendo troppi dati rispetto alla dimensione del buffer però il computer non sa la dimensione del buffer (per lui sono solo indirizzi di memoria).
Dunque continua a copiare `"stringatroppolungaperbuf"` a partire dal primo indirizzo di memoria di `buf[]`, fino a che non ha occupato tutti gli indirizzi di memoria del buffer, e poi continua a sovrascrivere qualsiasi cosa trovi, finché non completa l’operazione richiesta

### Evoluzione dello stack
Tra parentesi si trovano i valori che prendono i diversi indirizzi dello stack (semplificato)

![[Pasted image 20241216190346.png|center|400]]

In realtà un indirizzo conterrà più lettere, in base alla dimensione delle parole in memoria
![[Pasted image 20241216190501.png|400]]

