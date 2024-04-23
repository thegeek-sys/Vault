---
Created: 2024-04-23
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
Le eccezioni rappresentano un meccanismo utile a notificare e gestire gli errori e vengono generati quando durante l’esecuzione si è verificato un **errore**
Il termine “eccezione” indica un **comportamento anomalo**, che si discosta dalla normale esecuzione e impararle a gestire rende il codice più robusto e sicuro

Questa viene generata ad esempio quando provo ad accedere ad un elemento il cui indice non è presente in un array
```java
int[] estrazioneLotto = { 3, 29, 10, 23, 67 };
for (int i=0; i<=5; i++) System.out.println(estrazioneLotto[i]);

/*
Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException: 5
*/
```
In questo caso l’esecuzione viene interrotta e ci accorgiamo del superamento incontrollato dei confini dell’array

### Vantaggi
- In linguaggi come il C, la logica del programma e la logica di gestione degli errori sono **interlacciate**: questo rende più difficile leggere, modificare e mantenere il codice
- Gli errori vengono **propagati verso l’alto** lungo lo stack di chiamate
- Codice **più robusto**: non dobbiamo controllare esaustivamente tutti i possibili tipi di errore: il polimorfismo lo fa per noi, scegliendo l’intervento più opportuno

### Svantaggi
- L’**onere** di gestire i vari tipi di errore si sposta sulla JVM che si incarica di capire il modo più opportuno per gestire la situazione di errore

---
## Eccezioni notevoli

| Eccezione                    | Descrizione                                                                                                        |
| :--------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `IndexOutOfBoundsException`  | Accesso ad una posizione non valida di un array o una stringa (<0 o maggiore della sua dimensione)                 |
| `ClassCastException`         | Cast illecito di un oggetto ad una sottoclasse a cui non appartiene<br>Es. `Object x = new Integer(0); (Stringa)x` |
| `ArithmeticException`        | Condizione aritmetica non valida (es. divisione per zero)                                                          |
| `CloneNotSupportedException` | Metodo `clone()` non implementato o errore durante la copia dell’oggetto                                           |
| `ParseException`             | Errore inaspettato durante il parsing                                                                              |
| `IOError` e `IOException`    | Grave errore di input o output                                                                                     |
| `IllegalArgumentException`   | Parametro illegale come input di un metodo                                                                         |
| `NumberFormatException`      | Errore nel formato di un numero (estende la precedente)                                                            |

---
