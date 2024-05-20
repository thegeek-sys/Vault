---
Created: 2024-05-14
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
L’interfaccia `java.util.stream.Stream` che è stata pensata per **definire elaborazioni di flussi di dati**. La sua particolarità sta nel fatto che può supportare operazioni sequenziali e parallele.
Uno `Stream` viene creato a partire da una sorgente di dati, ad esempio una `java.util.Collection` ma al contrario delle collection, uno `Stream` **non memorizza né modifica i dati della sorgente**, ma opera su di essi

>[!info]- `java.util.Optional`
>![[Interfacce note#Optional< T>]]

---
## Operazioni
Le operazioni eseguibili dagli stream sono le operazioni intermedie e terminali
- **intermedie** → restituiscono un altro stream su cui continuare a lavorare (servono a dichiarare le operazioni da eseguire su uno stream)
- **terminali** → restituiscono il tipo atteso (ma una volta eseguita questa operazione lo stream non è più utilizzabile)

Inoltre si parla di **comportamento pigro** (lazy behavior), infatti le operazioni intermedie non vengono eseguite immediatamente, ma solo quando si richiede l'esecuzione di un'operazione terminale

Le operazioni possono essere:
- **senza stato** (*stateless*): l'elaborazione dei vari elementi può procedere in modo indipendente (es. `filter`). Nel in cui le operazioni siano tutte stateless permette alla JVM di scegliere l’ordine più efficienti per la loro esecuzione
- **con stato** (*stateful*): l'esecuzione delle operazioni intermedie dipende dagli elementi (es. `sorted`). Risultano bloccanti rispetto alla libertà di porter riordinare le operazioni

Poiché Stream opera su oggetti, esistono analoghe versioni ottimizzate per lavorare con 3 tipi primitivi:
- su **int** → `IntStream`
- su **double** → `DoubleStream`
- su **long** → `LongStream`

>[!hint] Tutte queste interfacce estendono l'interfaccia di base `BaseStream`

---
## Ottenere un o stream
Possiamo ottenere uno stream direttamente dai dati con il metodo statico generico `Stream.of(array di un certo tipo)`.
Da Java 8 l'interfaccia `Collection` è stata estesa per includere due nuovi metodi di default:
- **`default Stream<E> stream()`** → restituisce uno stream sequenziali
- **`default Stream<E> parallelStream()`** → restituisce un nuovo stream parallelo, se possibile (altrimenti restituisce uno stream sequenziale)

### Esempio
E' possibile ottenere uno stream anche per un array, con il metodo statico `Stream<T> Arrays.stream(T[ ] array)`
E' possibile ottenere uno stream di righe di testo da `BufferedReader.lines()` oppure da `Files.lines(Path)`
E’ possibile ottenere uno stream di righe anche da `String.lines`

---
## Stream vs. Collection
Lo stream permette di utilizzare uno **stile dichiarativo** con cui la JVM può decidere per motivi di efficienza di ordinare in modo differente le operazioni da eseguire. Mentre le collection obbligano l’utilizzo di uno **stile imperativo** in cui la JVM è obbligata a eseguire le operazioni nell’ordine in cui gli sono state proposte

---
## Metodi

| Metodo                                                                      | Tipo | Descrizione                                                                                         |
| --------------------------------------------------------------------------- | :--: | --------------------------------------------------------------------------------------------------- |
| `<R, A> R collect(Collector<? super T, A, R> collectorFunction)`            |  T   | Raccoglie gli elementi di tipo T in un contenitore di tipo R, accumulando in oggetti di tipo A      |
| `long count()`                                                              |  T   | Conta il numero di elementi                                                                         |
| `void forEach(Consumer<? super T> action)`                                  |  T   | Esegue il codice in input su ogni elemento dello stream                                             |
| `Optional<T> max/min(Comparator<? super T> comparator)`                     |  T   | Restituisce il massimo/minimo elemento all'interno dello stream                                     |
| `T reduce(T identityVal, BinaryOperator<T> accumulator)`                    |  T   | Effettua l'operazione di riduzione basata sugli elementi dello stream                               |
| `Stream<T> filter(Predicate<? super T> predicate)`                          |  I   | Fornisce uno stream che contiene solo gli elementi che soddisfano il predicato                      |
| `<R> Stream<R> map(Function<? super T, ? extends R> mapFunction)`           |  I   | Applica la funzione a tutti gli elementi dello stream, fornendo un nuovo stream di elementi mappati |
| `IntStream mapToInt/ToDouble/ToLong( ToIntFunction<? super T> mapFunction)` |  I   | Come sopra, ma la mappatura è su interi/ecc. (**operazione ottimizzata**)                           |
| `Stream<T> sorted()`                                                        |  I   | Produce un nuovo stream di elementi ordinati                                                        |
| `Stream<T> limit(long k)`                                                   |  I   | Limita lo stream a k elementi                                                                       |
### min e max
I metodi `min` e `max` restituiscono rispettivamente il minimo e il massimo di uno stream sotto forma di `Optional`. Prendono in input un `Comparator` sul tipo degli elementi dello stream
```java
List<Integer> p = Arrays.asList(2,3,4,5,6,7);
Optional<Integer> max = p.stream().max(Integer::compare);

// se c'è, restituisce il massimo; altrimenti restituisce -1
System.out.println(max.orElse(-1));
```

### filter (intermedio) forEach (terminale)
`filter` è un metodo di `Stream` che accetta un predicato (`Predicate`) per filtrare gli elementi dello stream e **restituisce lo stream filtrato**
`forEach` prende in input un `Consumer` e lo applica a ogni elemento dello stream (operazione terminale)

#### Esempio
Filtra gli elementi di una lista di interi mantenendo solo quelli dispari e stampa ciascun elemento rimanente:
```java
List<Integer> l = Arrays.asList(4, 8, 15, 16, 23, 42);
l.stream()
 .filter(k -> k % 2 == 1)
 .forEach(System.out::println);
```

Filtra gli elementi di una lista per iniziale e lunghezza della stringa e stampa ciascun elemento rimanente:
```java
Predicate<String> startsWithJ = s -> s.startsWith("J");
Predicate<String> fourLetterLong = s -> s.length() == 4;

List<String> l = Arrays.asList("Java", "Scala", "Lisp");
l.stream()
 .filter(startsWithJ.and(fourLetterLong))
 .forEach(s -> System.out.println("Inizia con J ed e’ lungo 4 caratteri: "+s);
```

### count (terminale)
`count` è un’operazione terminale che restituisce il numero `long` di elementi nello stream

#### Esempio
```java
long startsWithA = l.stream().filter(s -> s.startsWith("a")) .count();
System.out.println(startsWithA); // 2
```

Conteggio del numero di righe di un file di testo:
```java
long numberOfLines = Files.lines(Paths.get("yourFile.txt")).count();
```

### sorted (intermedia)
`sorted` è un’operazione intermedia sugli stream che restituisce una vista ordinata dello stream senza modificare la collezione sottostante

#### Esempio
```java
// in qeusto caso la JVM decide prima di filtrare lo stream per poi
// ordinarlo, nonostante abbia programmato il contrario
List<String> l = Arrays.asList("da", "ac", "ab", "bb");
l.stream()
 .sorted()
 .filter(s -> s.startsWith("a"))
 .forEach(System.out::println);

// ab, ac
```

### map (intermedia)
`map` è un’operazione intermedia sugli stream che restituisce un nuovo stream in cui ciascun elemento dello stream di origine è convertito in un altro oggetto attraverso la funzione (`Function`) passata in input

#### Esempio
Restituire tutte le stringhe (portate in maiuscolo) ordinate in ordine inverso
```java
// equivalente a .map(s -> s.toUpperCase())
l.stream().map(String::toUpperCase)
          .sorted(Comparator<String>.naturalOrder().reversed())
          .forEach(System.out::println);
// DA, BB, AC, AB
```

Si vuole scrivere un metodo che aggiunga l’IVA a ciascun prezzo:
```java
List<Integer> ivaEsclusa = Arrays.asList(10, 20, 30);

// In Java 7:
for (int p : ivaEsclusa) {
	double pIvaInclusa = p*1.22;
	System.out.println(pIvaInclusa);
}

// In Java 8:
ivaEsclusa.stream()
		  .map(p -> p*1.22)
		  .forEach(System.out::println);
```

### collect (terminale)
`collect` è un’operazione terminale che permette di raccogliere gli elementi dello stream in un qualche oggetto (ad es. una collection, una stringa, un intero)

#### Esempio
Ottenere la lista dei prezzi ivati:
```java
List<Integer> ivaEsclusa = Arrays.asList(10, 20, 30);

// In Java 7:
List<Double> l = new ArrayList<>();
for (int p : ivaEsclusa) l.add(p*1.22);

// In Java 8:
List<Double> l = ivaEsclusa.stream().map(p -> p*1.22)
						   .collect(Collectors.toList());
```

Creare una stringa che concatena stringhe in un elenco, rese maiuscole e separate da virgola:
```java
List<String> l = Arrays.asList("RoMa", "milano", "Torino");
String s = "";

// in Java 7:
for (String e : l) s += e.toUpperCase()+", ";
s = s.substring(0, s.length()-2);

// in Java 8:
s = l.stream().map(String::toUpperCase)
			  .collect(Collectors.joining(", "));
```

Trasformare una lista di stringhe in una lista delle lunghezze delle stesse:
```java
List<String> words = Arrays.asList("Oracle", "Java", "Magazine");
List<Integer> wordLengths =
	words.stream()
		 .map(String::length)
		 .collect(toList());
```

---
## Collectors
I `Collectors` sono delle “ricette” per **ridurre gli elementi di uno stream** e raccoglierli in qualche modo
Per rendere più leggibile il codice: `import static java.util.stream.Collectors.*` (importo tutti i metodi statici di `Collectors`)
- In questo modo possiamo scrivere il nome del metodo senza anteporre Collectors. (es. `toList()` invece di `Collectors.toList()`)

### Riduzione a singolo elemento
**`counting()`** restituisce un `Collector` che ha come unico elemento il numero elementi di uno stream (risultato di tipo long)
```java
List<Integer> l = Arrays.asList(2, 3, 5, 6);
// k == 2
long k = l.stream()
		  .filter(x -> x < 5)
		  .collect(Collectors.counting()));
```

**`maxBy/minBy(comparator)`** restituisce un `Optional` con il massimo/minimo valore
```java
// max contiene 6
Optional<Integer> max = l.stream()
						 .collect(maxBy(Integer::compareTo));
```

**`joining()`**, `joining(separatore)`, `joining(separatore, prefisso, suffisso)` concatena gli elementi stringa dello stream in un'unica stringa finale
```java
List<Integer> l = Arrays.asList(2, 3, 5, 6, 2, 7);
// str.equals("2,3,5,6,2,7")
String str = l.stream().map(x -> ""+x).collect(joining(","));
```

**`toList`**, **`toSet`** e **`toMap`** accumulano gli elementi in una lista, insieme o mappa (non c'è garanzia sul tipo di `List`, `Set` o `Map`)
```java
Set<String> set = l.stream().map(x -> ""+x).collect(toSet());
```

**`toCollection`** accumula gli elementi in una collezione scelta
```java
ArrayList<String> str = l.stream().map(x -> ""+x)
						.collect(toCollection(ArrayList::new));
```

### Riduzione a mappa
`toMap` prende in input fino a 4 argomenti:
- la funzione per mappare l'oggetto dello stream nella chiave della mappa
- la funzione per mappare l'oggetto dello stream nel valore della mappa
- *opzionale*: la funzione da utilizzare per unire il valore preesistente nella mappa a fronte della chiave con il valore associato all'oggetto dalla seconda funzione (non devono trovarsi due chiavi uguali o si ottiene un'eccezione `IllegalStateException`)
- *opzionale*: il Supplier che crea la mappa (se voglio utilizzare tipi diversi di mappa)

```java
Map<Integer, String> map = persons.stream()
								  .collect(Collectors.toMap
									  Person::getAge,
									  Person::getName,
									  (name1, name2) -> name1 + ";" name2));
```

### Raggruppamento di elementi
Per creare una **mappa tra chiavi di un tipo e lista di oggetti di un altro** tipo dovrò usare `groupingBy` che prende in input una lambda function che mappa gli elementi di tipo `T` in bucket rappresentati da oggetti di qualche altro tipo S e restituisce `Map<S, List<T>>`
Utilizzo invece `groupingBy(lambda, downStreamCollector)`, per raggruppamento multilivello

Ottenere una mappa da città a lista di persone a partire da una lista di persone:
```java
// people è una collection di Person
Map<City, List<Person>> peopleByCity = people.stream()
									  .collect(groupingBy(Person::getCity));
```

La stessa mappa, ma i cui valori siano insiemi di persone:
```java
Map<City, Set<Person>> peopleByCity = people.stream()
							  .collect(goupingBy(Person::getCity), toSet());
```

#### mapping
In raccolte multilivello, per esempio usando groupingBy, è utile mappare il valore del raggruppamento a qualche altro tipo:
```java
Map<City, Set<String>> peopleSurnamesByCity =
	people.stream().collect(
		groupingBy(Person::getCity,
			mapping(Person::getLastName, toSet())));
```

### Creare il proprio Collector
Con il metodo statico `Collector.of` che prende in input 4 argomenti:
- un **supplier** → per creare la rappresentazione interna
- l’**accumulator** → che aggiorna la rappresentazione con il nuovo metodo
- un **combiner** → che “fonde” due rappresentazioni ottenute in modo parallelo
- il **finisher** → chiamato alla fine, che trasforma tutto nel tipo finale

```java
Colector<Person, StringJoiner, String> personNameCollector =
	Collector.of(
		() -> new StringJoiner(" | "), // supplier
		
		// j è lo StringJoiner
		(j, p) -> j.add(p.name.toUpperCase()), // accumulator
		
		// fa il merge di tutti gli StringJoiner pronti dato
		// che implicitamente si hanno molti StringJoiner
		// eseguiti in parallelo
		(j1, j2) -> j1.merge(j2), // combiner
		
		StringJoiner::toString // finisher
	);
	
	String names = people.stream()
						 .collect(personNmaeCollector);
	System.out.println(names); // MAX | PETER | PAMELA
```

### Partizionamento di elementi
