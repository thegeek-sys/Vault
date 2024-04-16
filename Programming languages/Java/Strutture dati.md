---
Created: 2024-04-15
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
	- [[#Introduction#Caratteristiche|Caratteristiche]]
- [[#Collection|Collection]]
	- [[#Collection#Iterazione esterna su una collezione|Iterazione esterna su una collezione]]
	- [[#Collection#Iterazione interna su una collezione|Iterazione interna su una collezione]]
- [[#Modificare una lista durante l'iterazione|Modificare una lista durante l'iterazione]]
- [[#Collezioni fondamentali|Collezioni fondamentali]]
- [[#ArrayList e LinkedList|ArrayList e LinkedList]]
	- [[#ArrayList e LinkedList#Metodi ArrayList|Metodi ArrayList]]
	- [[#ArrayList e LinkedList#Alcuni metodi aggiuntivi di LinkedList|Alcuni metodi aggiuntivi di LinkedList]]
	- [[#ArrayList e LinkedList#Iterare sulle list in entrambe le direzioni|Iterare sulle list in entrambe le direzioni]]
- [[#Insiemi: HashSet, TreeSet e LinkedHashSet|Insiemi: HashSet, TreeSet e LinkedHashSet]]
	- [[#Insiemi: HashSet, TreeSet e LinkedHashSet#Esempio: HashSet|Esempio: HashSet]]
	- [[#Insiemi: HashSet, TreeSet e LinkedHashSet#Esempio: TreeSet|Esempio: TreeSet]]
- [[#Mappe|Mappe]]
	- [[#Mappe#Metodi|Metodi]]
	- [[#Mappe#HashMap|HashMap]]
		- [[#HashMap#Esempio|Esempio]]
	- [[#Mappe#TreeMap|TreeMap]]
		- [[#TreeMap#Esempio|Esempio]]
	- [[#Mappe#LinkedHashMap|LinkedHashMap]]
		- [[#LinkedHashMap#Esempio|Esempio]]
	- [[#Mappe#Confronto tra HashMap, TreeMap e LinkedHashMap|Confronto tra HashMap, TreeMap e LinkedHashMap]]
	- [[#Mappe#Chiavi e valori di una mappa|Chiavi e valori di una mappa]]
		- [[#Chiavi e valori di una mappa#Esempio|Esempio]]
	- [[#Mappe#Metodi Java 8 e Java 9 nell’interfaccia java.util.Map|Metodi Java 8 e Java 9 nell’interfaccia java.util.Map]]
- [[#Algoritmi sulle collezioni: la classe java.util.Collection|Algoritmi sulle collezioni: la classe java.util.Collection]]
- [[#Algoritmi sugli array: la classe java.util.Arrays|Algoritmi sugli array: la classe java.util.Arrays]]
- [[#Ordinamento naturale|Ordinamento naturale]]
	- [[#Ordinamento naturale#Esempio|Esempio]]
- [[#Ordinamento con l’interfaccia Comparator|Ordinamento con l’interfaccia Comparator]]

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
Il metodo `listIterator()` restituisce un **iteratore bidirezionale** per la lista

Da sinistra verso destra
```java
ListIterator<Integer> i = l.listIterator();
while(i.hasNext())
	System.out.println(i.next());
```

Da destra verso sinistra (se si specifica un intero, si parte da quella posizione)
```java
ListIterator<Integer> i = l.listIterator(l.size());
while(i.hasPrevious())
	System.out.println(i.previous());
```

---
## Insiemi: HashSet, TreeSet e LinkedHashSet
Gli **insiemi** sono basati su `Set`, una sottointerfaccia di `Collection` e di `Iterable`. Proprio come gli insieme matematici anche questi contengono elementi **tutti distinti**.

**HashSet** memorizza gli elementi di una tabella di hash. Questo si fonda sul concetto di **tabella hash** 
$$
\text{se a.eqauls(b)} \longrightarrow (\text{a.hashCode}== \text{b.hashCode})
$$
$$
(\text{a.hashCode}== \text{b.hashCode}) \centernot\longrightarrow \text{se a.eqauls(b)}
$$


**TreeSet** memorizza gli elementi in un albero mantenendo un ordine sugli elementi (ordinamento naturale dei tipi)

**LinkedHashSet** memorizza gli elementi in ordine di inserimento

### Esempio: HashSet
```java
HashSet<String> nomi = new HashSet<String>();
HashSet<String> cognomi = new HashSet<String>();

nomi.add("mario");
cognomi.add("rossi");

nomi.add("mario");
cognomi.add("verdi");

nomi.add("luigi");
cognomi.add("rossi");

nomi.add("luigi");
cognomi.add("bianchi");

System.out.println(nomi);  // [mario, luigi]
System.out.println(cognomi);  // [verdi, bianchi, rossi]
```

### Esempio: TreeSet
Gli elementi vengono mantenuti ordinati sulla base dell’ordinamento naturale definito sul tipo degli elementi
```java
TreeSet<String> nomi = new TreeSet<String>();
TreeSet<String> cognomi = new TreeSet<String>();

nomi.add("mario");
cognomi.add("rossi");

nomi.add("mario");
cognomi.add("verdi");

nomi.add("luigi");
cognomi.add("rossi");

nomi.add("luigi");
cognomi.add("bianchi");

System.out.println(nomi);  // [luigi, mario]
System.out.println(cognomi);  // [bianchi, rossi, verdi]
```

---
## Mappe
Una mappa mette in corrispondenza **chiavi** e **valori** e non può contenere chiavi duplicate.
`java.util.Map` è un’interfaccia implementata da `HashMap`, `LinkedHashMap`, `TreeMap`

### Metodi

| Tipo                  | Method                                                                                                                                  |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `void`                | **`clear()`**<br>Removes all of the mappings from this map                                                                              |
| `boolean`             | **`containsKey(Object key)`**<br>Returns true if this map contains a mapping for the specified key.                                     |
| `boolean`             | **`containsValue(Object value)`**<br>Returns true if this map maps one or more keys to the specified value.                             |
| `Set<Map.Entry<K,V>>` | **`entrySet()`**<br>Returns a `Set` view of the mappings contained in this map.                                                         |
| `boolean`             | **`equals(Object o)`**<br>Compares the specified object with this map for equality.                                                     |
| `V`                   | **`get(Object key)`**<br>Returns the value to which the specified key is mapped, or `null` if this map contains no mapping for the key. |
| `int`                 | **`hashCode()`**<br>Returns the hash code value for this map.                                                                           |
| `boolean`             | **`isEmpty()`**<br>Returns true if this map contains no key-value mappings.                                                             |
| `Set<K>`              | **`keySet()`**<br>Returns a `Set` view of the keys contained in this map                                                                |
| `V`                   | **`put(K key, V value)`**<br>Associates the specified value with the specified key in this map                                          |
| `void`                | **`putAll(Map<? extends K,? extends V>m)`**<br>Copies all of the mappings from the specified map to this map                            |
| `V`                   | **`remove(Object key)`**<br>Removes the mapping for a key from this map if it is present                                                |
| `int`                 | **`size()`**<br>Returns the number of key-value mappings in this map.                                                                   |
| `Collection<V>`       | **`values()`**<br>Returns a `Collection` view of the values contained in this map                                                       |


### HashMap
Memorizza le coppie in una tabella hash
#### Esempio
Registriamo le frequenze di occorrenza delle parole in un testo
```java
public class MappaDelleFrequenze {
	private Map<String, Integer> frequenze = new HashMap<String, Integer>;
	public MappaDelleFrequenze(File file) throws IOException {
		Scanner in = new Scanner(file);
		
		while(in.hasNext()) {
			Singola parola = in.next();
			Integer freq = frequenze.get(parola);
			
			// parola mai incontrata (non presente nella mappa)
			if (freq == null)  frequenze.put(parola, 1);
			// incrementa il conteggio di frequenza della parola
			else               frequenze.put(parola, freq+1);
		}
	}
}
```

### TreeMap
Memorizza le coppie in un albero mantenendo un ordine sulle chiavi
#### Esempio
Se vogliamo mantenere un ordinamento sulle chiavi (ovvero sulle parole) di cui calcoliamo la frequenza
```java
public class MappaDelleFrequenze {
	private Map<String, Integer> frequenze = new TreeMap<String, Integer>();
	
	public MappaDelleFrequenze(File file) throws IOException {
		Scanner in = new Scanner(file);
		
		while(in.hasNext()) {
			Singola parola = in.next();
			Integer freq = frequenze.get(parola);
			
			// parola mai incontrata (non presente nella mappa)
			if (freq == null)  frequenze.put(parola, 1);
			// incrementa il conteggio di frequenza della parola
			else               frequenze.put(parola, freq+1);
		}
	}
}
```

### LinkedHashMap
Estende HashMap e mantiene l’ordinamento di iterazione secondo gli inserimenti effettuati
#### Esempio
Se vogliamo mantenere un ordinamento di iterazione (ovvero sulle parole) di cui calcoliamo la frequenza
```java
public class MappaDelleFrequenze {
	private Map<String, Integer> frequenze = new LinkedHashMap<String, Integer>();
	
	public MappaDelleFrequenze(File file) throws IOException {
		Scanner in = new Scanner(file);
		
		while(in.hasNext()) {
			Singola parola = in.next();
			Integer freq = frequenze.get(parola);
			
			// parola mai incontrata (non presente nella mappa)
			if (freq == null)  frequenze.put(parola, 1);
			// incrementa il conteggio di frequenza della parola
			else               frequenze.put(parola, freq+1);
		}
	}
}
```

### Confronto tra HashMap, TreeMap e LinkedHashMap
Dato il seguente testo:
`a a e b c d e e e b`

Se iteriamo su ciascuna delle tre strutture otteniamo:
- **HashMap** (senza ordinamento evidente) → `{d=1, e=3, b=2, c=1, a=2}`
- **TreeMap** (ordinamento naturale) → `{a=2, b=2, c=1, d=1, e=3}`
- **LinkedHashMap** (ordinamento di inserimento) → `{a=2, e=3, b=2, c=1, d=1}`

### Chiavi e valori di una mappa
E’ possibile ottenere l’insieme delle chiavi di una mappa mediante il metodo `keySet` e l’elenco dei valori mediante il metodo `values` (con ripetizione!)
L’insieme delle coppie (chiave, valore) mediante il metodo `entrySet` che:
- restituisce un insieme di oggetti di tipo `Map.Entry(K, V)`
- per ogni elemento (ovvero la coppia) è possibile conoscere la chiave (`getKey()`) e il valore (`getValue()`)

#### Esempio
Il seguente codice crea una **TreeMap** e ne stampa l’insieme delle chiavi, la collezione dei valori e la collezione delle coppie (di tipo `Map.Entry<K,V>`)
```java
Map<String, Integer> m = new TreeMap<String, Integer>();

m.put("a", 1);
m.put("b", 1);
m.put("c", 2);
m.put("d", 3);
m.put("e", 1);

System.out.println(m.keySet());   // [a, b, c, d, e]
System.out.println(m.values());   // [1, 1, 2, 3, 1]
System.out.println(m.entrySet()); // [a=1, b=1, c=2, d=3, e=1]
```

### Metodi Java 8 e Java 9 nell’interfaccia java.util.Map
`forEach(BiConsumer)`
Itera su ciascuna coppia (chiave, valore)

`getOrDefault(chiave, valoreDefault)`
Restituisce il valore associato alla chiave o valoreDefault se la chiave non è presente

`merge(chiave, valore, BiFunction)`
Se la chiave non contiene già un valore, imposta il valore specificato, altrimenti chiama una bifunzione che decide come mettere insieme il valore precedente con il valore passato in input

`of(chiave, valore, chiave, valore, ..., chiave, valore)`
Statico, crea una mappa immutabile dei tipi e con i valori corrispondenti

---
## Algoritmi sulle collezioni: la classe java.util.Collection
La classe `java.util.Collection` fornisce metodi **statici** per la manipolazione delle collezioni

| Metodo         | Descrizione                                                         |
| -------------- | ------------------------------------------------------------------- |
| `sort`         | Ordina gli elementi di una List                                     |
| `binarySearch` | Cerca un elemento di una List mediante ricerca binaria              |
| `fill`         | Rimpiazza tutti gli elementi di una List con l’elemento specificato |
| `copy`         | Copia gli elementi da una lista all’altra                           |
| `reverse`      | Inverte l’ordine degli elementi di una List                         |
| `shuffle`      | Mette in ordine casuale gli elementi di una List                    |
| `min/max`      | Restituisce l’elemento più piccolo/grande della Collection          |

---
## Algoritmi sugli array: la classe java.util.Arrays
La classe `java.util.Arrays` fornisce metodi **statici** per la manipolazione degli array

| Metodo         | Descrizione                                                        |
| -------------- | ------------------------------------------------------------------ |
| `sort`         | Ordina gli elementi dell’array                                     |
| `binarySearch` | Cerca un elemento mediante ricerca binaria                         |
| `fill`         | Riempi l’array con l’elemento specificato                          |
| `copyOf`       | Restituisce una copia di un array                                  |
| `equals`       | Confronta due array elemento per elemento                          |
| `asList`       | Restituisce una lista contenente gli elementi dell’array           |
| `toString`     | Restituisce una rappresentazione dell’array sotto forma di stringa |

---
## Ordinamento naturale
Come **garantire un ordinamento sui tipi** utilizzati nelle strutture dati che si fondano su un ordinamento (es. `TreeSet` o `TreeMap`)?
E’ necessario che quei tipi implementino un’interfaccia speciale, chiamata **`Comparable<T>`** dotata di un solo metodo

| Metodo               | Descrizione                                                                             |
| :------------------- | :-------------------------------------------------------------------------------------- |
| `int compareTo(T o)` | Confronta se stesso con l’oggetto o (restituendo 0 se uguali, -1 se ≤ o, +1 altrimenti) |

### Esempio
```java
public class NomeCognome implements Comparable<NomeCognome> {
	private String nome;
	private String cognome;
	
	public NomeCognome(String nome, String cognome) {
		this.nome = nome;
		this.cognome = cognome;
	}
	
	@Override
	public int compareTo(NomeCognome o) {
		int v = cognome.compareTo(o.cognome);
		if (v == 0) return nome.compareTo(o.nome);
		else return v
	}
	
	@Override
	public String toString() { return cognoome+" "+nome; }

}
```

---
## Ordinamento con l’interfaccia Comparator
E se la classe **non fornisce “di suo“ l’ordinamento** mediante l’interfaccia Comparable? E se vogliamo **ordinare diversamente** gli elementi di un certo oggetto?
E’ sufficiente implementare un’interfaccia **`Comparator<T>`** e passarne un’istanza in input al costruttore delle strutture dati (es. TreeSet e TreeMap)

Metodi dell’interfaccia `Comparator<T>`

| Tipo      | Metodo                                                                                        |
| --------- | --------------------------------------------------------------------------------------------- |
| `int`     | **`compare(T o1, T o2)`**<br>Compares its two arguments for order                             |
| `boolean` | **`equals(Object obj)`**<br>Indicates whether some other object is “equal to” this comparator |
