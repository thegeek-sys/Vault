---
Created: 2024-04-15
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
Una struttura dati serve a memorizzare e organizzare i dati in memoria in modo tale da poterli usare efficientemente

### Caratteristiche
Per poter scegliere una struttura dati è necessario chiedersi:
1. E’ necessario mantenere un ordine?
2. Gli oggetti nella struttura possono ripetersi?
3. E’ utile/necessario possedere una “chiave” per accedere a uno specifico oggetto?

---
## Collection
Le collezioni in Java sono rese disponibili mediante il *framework delle collezioni* (Java Collection Framework)
Queste sono strutture dati già pronte all’uso, disponendo all’utente **interfacce** e **algoritmi** per manipolarle

Contengono e “strutturano” **riferimenti** ad altri oggetti (tipicamente tutti “dello stesso tipo“)

| Interfaccia    | Descrizione                                                |
| :------------- | :--------------------------------------------------------- |
| **Collection** | L’interfaccia alla radice della gerarchia di collezioni    |
| **Set**        | Una collezione senza duplicati                             |
| **List**       | Una collezione ordinata che può contenere duplicati        |
| **Map**        | Associa coppie di (chiave, valore), senza chiavi duplicate |
| **Queue**      | Una collezione first-in, first-out che modella una coda    |

![[Screenshot 2024-04-11 alle 13.04.04.png|In giallo i metodi astratti che le interfacce mettono a disposizione per le classi che le implementano]]


### Iterazione esterna su una collezione

- mediante gli **iterator** → “vecchio stile” ma maggiore controllo
```java
Iterator<Integer> i = collezione.iterator();
while(i.hasNext()) {      // finché ha un successivo
	int k = i.next();     // ottieni prossimo elemento
	System.out.println(k);
}
```

- mediante il costrutto “**for each**”
```java
for (Integer k : collezione)
	System.out.println(k);
```

- mediante gli **indici**
```java
for (int j=0; j<collezione.size(); j++) {
	int k = collezione.get(j);
	System.out.println(k);
}
```

### Iterazione interna su una collezione
Mediante il metodo `Iterable.forEach` permette l’iterazione su qualsiasi collezione senza specificare come effettuare l’iterazione (utilizza il polimorfismo, chiamerà il forEach della classe specifica)
**forEach** prende in input un *Consumer*, che è un’interfaccia funzionale con un solo metodo
`void accept(T t);`

Esempio:
```java
List<Integer> I = List.of(4, 8, 15, 16, 23, 42)
I.forEach(x -> System.out.println(x));
```

> [!hint] [.forEach vs. for-each loop](https://stackoverflow.com/questions/16635398/java-8-iterable-foreach-vs-foreach-loop)

---
## Modificare una lista durante l'iterazione
Non è possibile utilizzare metodi di modifica durante un'iterazione:
```java
for (int k : l)
	if (k == x) { l.remove(x); }

// java.util.CurrentModificationException
```

ma è possibile utilizzare `Iterator.remove`:
```java
Iterator<Integer> i = l.iterator();
while(i.hasNext())
	if (i.next() == x) i.remove();
```

---
## Collezioni fondamentali
- AbstractList
	- **ArrayList**
	- **LinkedList**
- AbstractSet
	- **HashSet**
	- **TreeSet**
- HashSet
	- **LinkedHashSet**
- AbstractMap
	- **HashMap**
	- **TreeMap**
- HashMap
	- **LinkedHashMap**