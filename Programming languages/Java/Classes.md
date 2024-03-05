---
Created: 2024-03-05
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index

- [[#Introduzione|Introduzione]]
- [[#Classe e oggetto a confronto|Classe e oggetto a confronto]]
- [[#Esempio pratico|Esempio pratico]]
- [[#File sorgenti|File sorgenti]]
- [[#Struttura|Struttura]]
- [[#Esercizio: contatore|Esercizio: contatore]]
---
## Introduzione
Una classe è una parte del codice che fornisce un prototipo astratto per gli oggetti di un particolare tipo.
Un programma può creare e usare uno o più oggetti (istanze) della stessa classe.

Quando definisco una classe descrivo solo le caratteristiche di questa classe (campi e metodi) e nell’implementazione definisco gli oggetti della classe con relativi campi e metodi (es. classe: automobile, oggetto: toyota).

In conclusione la classe **specifica la struttura di un oggetto** (dei campi dei suoi oggetti) **e il comportamento dei suoi oggetti** mediante i metodi; mentre l’oggetto contiene **specifici valori dei campi** (che possono cambiare durante l’esecuzione)

---
## Classe e oggetto a confronto

| Classe:                                                                   | Oggetto:                                                                                                          |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Definita mediante parte del codice sorgente del programma                 | Un’entità all’interno di un programma in esecuzione                                                               |
| Scritta dal programmatore                                                 | Creato quando un programma “gira” (dal metodo main o da un altro metodo)                                          |
| Specifica la struttura (ovvero numero e tipi) dei campi dei suoi oggetti  | Contiene specifici valori dei campi; i valori possono cambiare durante l’esecuzione                               |
| Specifica il comportamento dei suoi oggetti mediante il codice dei metodi | Si comporta nel modo prescritto dalla classe quando il metodo corrispondente viene chiamato a tempo di esecuzione |

---
## Esempio pratico

![[sonic.png|center|200]]
In questo caso esiste una classe anello da cui posso creare tutte le istanze (oggetti) anello che mi servono
```
classe Anello
- Campi:
	- int x
	- int y
	- int r
- Metodi
	- ruota
```

---
## File sorgenti
Ogni classe è memorizzata in un file separato e il nome del file DEVE essere lo stesso della classe. con estensione `.java`.
I nomi di classe inoltre iniziano sempre con una maiuscola (es. Automobile)

> [!warning]
> I nomi in Java sono case-sensitive

```java
'''
Automobile.java
'''

public class Automobile {
	...
}
```

---
## Struttura

![[Screenshot 2024-03-05 alle 15.31.31.png]]

---
## Esercizio: contatore
Implemento una classe contatore che mi permette di:
- **incrementare** il conteggio attuale
- **ottenere** il conteggio attuale
- **resettare** il conteggio a `0` o ad un numero a mia scelta

```java
public class Counter {  
    private int value;  
	
	
	/* Costruttore della classe (es. main)
	** (stesso nome della classe e non ha tipo di ritorno)
	*/
    public Counter() {  
        value = 0;  
    }
    public Counter(int initValue) { // overloading in modo da poter
        value = initValue;          // usare un valore iniziale a
    }                               // mia scelta
	
	
	/* Metodi della classe
	** (è void perché non restituisce nulla)
	*/
    public void count() {  
        value++;  
    }  
    public void reset() {  
        value = 0;  
    }  
    public void reset(int newValue) {  // overloading del metodo per
		value = newValue;   		   // resettare ad un
    }                                  // determinato valore
	
	
	/* Metodo "getter" che restituisce un intero */
	public int getValue() { return value; }
}
```