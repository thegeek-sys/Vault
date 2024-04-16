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

---
## ArrayList e LinkedList
**ArrayList** e **LinkedList** sono due strutture dati sono basate su `List` (la implementano), una sottointerfaccia di `Collection` e di `Iterable`, estendendo `AbstractList`

**ArrayList** implementa una lista mediante un **array** (eventualmente ridimensionato, la sua capacità iniziale è 10 elementi)
**LinkedList** implementa la lista mediante **elementi linkati**

### Metodi ArrayList

|             Tipo | Metodo                                                                                                                                                                                                      |
| ----------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|        `boolean` | **`add(E e)`**<br>Appends the specific element to the end of the list                                                                                                                                       |
|           `void` | **`add(int index, E element)`**<br>Inserts the specific element at the specific position in the list                                                                                                        |
|        `boolean` | **`addAll(Collection<? extends E> c)`**<br>Appends all of the elements in the specified collection to the end of this list, in the order that they are returned by the specified collection’s Iterator      |
|         `boolan` | **`addAll (int index, Collection<? extends E> c)`**<br>Inserts all of the elements in the specified collection into this list, starting at the specified position                                           |
|           `void` | **`clear()`**<br>Removes all of the elements from this list                                                                                                                                                 |
|         `Object` | **`clone()`**<br>Returns a shallow copy of this `ArrayList` instance                                                                                                                                        |
|        `boolean` | **`contains(Object o)`**<br>Returns `true` if this list contains the specified element                                                                                                                      |
|           `void` | **`ensureCapacity(int minCapacity)`**<br>Increases the capacity of this `ArrayList` instance, if necessary, to ensure that it can hold at least the number of elements specified by the minimum             |
|              `E` | **`get(int index)`**<br>Return the element at  the specified position in this list                                                                                                                          |
|            `int` | **`indexOf(Object o)`**<br>Returns the index of the first occurrence of the specified element in this list, or -1 if this list does not contain the element                                                 |
|        `boolean` | **`isEmpty()`**<br>Returns `true` if this list contains no elements                                                                                                                                         |
|            `int` | **`lastIndexOf(Object o)`**<br>Returns the index of the last occurrence of the specified element in this list, or -1 if this list does not contain the element                                              |
|              `E` | **`remove(int index)`**<br>Removes the element at the specified position in this list                                                                                                                       |
|        `boolean` | **`remove(Object o)`**<br>Removes the first occurrence of the specified element from this list, if it’s present                                                                                             |
| `protected void` | **`removeRange(int fromIndex, int toIndex)`**<br>Removes from this list all of the elements whose index is between `fromIndex` inclusive, and `toIndex` exclusive                                           |
|              `E` | **`set(int index, E element)`**<br>Replaces the element at the specified position in this list with the specified element                                                                                   |
|            `int` | **`size()`**<br>Returns the number of elements in this list                                                                                                                                                 |
|       `Object[]` | **`toArray()`**<br>Returns an array containing all of the elements in this list in proper sequence (from first to last element)                                                                             |
|        `<T> T[]` | **`toArray(T[] a)`**<br>Returns an array containing all of the elements in this list in proper sequence (from first to last element); the runtime type of the returned array is that of the specified array |
|           `void` | **`trimToSize()`**<br>Trims the capacity of this `ArrayList` instance to be the list’s current size                                                                                                         |

### Alcuni metodi aggiuntivi di LinkedList

| Metodo                             | Descrizione                                                                                   |
| :--------------------------------- | :-------------------------------------------------------------------------------------------- |
| `void addFirst(E e)`               | Aggiungere l’elemento in testa alla lista                                                     |
| `void addLast(E e)`                | Aggiungere l’elemento in coda alla lista                                                      |
| `Iterator<E> descendingIterator()` | Restituire un iteratore che parte dall’ultimo elemento della lista e si sposta verso sinistra |
| `E getFirst()`                     | Restituisce il primo elemento della lista                                                     |
| `E getLast()`                      | Restituisce l’ultimo elemento della lista                                                     |
| `E removeFirst()`                  | Rimuove e restituisce il primo elemento                                                       |
| `E removeLast()`                   | Rimuove e restituisce l’ultimo elemento                                                       |
| `E pop()`                          | Elimina e restituisce l’elemento in cima alla lista vista come pila                           |
| `void push(E e)`                   | Inserisce un elemento in cima alla lista vista come pila                                      |

### Iterare sulle list in entrambe le direzioni