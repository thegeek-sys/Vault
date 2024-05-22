---
Created: 2024-05-14
Programming language: "[[Java]]"
Related: 
Completed:
---
---

## Index
- [[#Introduction|Introduction]]
- [[#Operazioni|Operazioni]]
- [[#Ottenere un o stream|Ottenere un o stream]]
	- [[#Ottenere un o stream#Esempio|Esempio]]
- [[#Stream vs. Collection|Stream vs. Collection]]
- [[#Metodi|Metodi]]
	- [[#Metodi#min e max|min e max]]
	- [[#Metodi#filter (intermedio) forEach (terminale)|filter (intermedio) forEach (terminale)]]
		- [[#filter (intermedio) forEach (terminale)#Esempio|Esempio]]
	- [[#Metodi#count (terminale)|count (terminale)]]
		- [[#count (terminale)#Esempio|Esempio]]
	- [[#Metodi#sorted (intermedia)|sorted (intermedia)]]
		- [[#sorted (intermedia)#Esempio|Esempio]]
	- [[#Metodi#map (intermedia)|map (intermedia)]]
		- [[#map (intermedia)#Esempio|Esempio]]
	- [[#Metodi#collect (terminale)|collect (terminale)]]
		- [[#collect (terminale)#Esempio|Esempio]]
	- [[#Metodi#distinct (intermedia)|distinct (intermedia)]]
		- [[#distinct (intermedia)#Esempio|Esempio]]
	- [[#Metodi#reduce (terminale)|reduce (terminale)]]
		- [[#reduce (terminale)#Esempio|Esempio]]
	- [[#Metodi#limit (intermedia)|limit (intermedia)]]
		- [[#limit (intermedia)#Esempio|Esempio]]
	- [[#Metodi#skip (intermedia)|skip (intermedia)]]
		- [[#skip (intermedia)#Esempio|Esempio]]
	- [[#Metodi#takeWhile/dropWhile (intermedie)|takeWhile/dropWhile (intermedie)]]
		- [[#takeWhile/dropWhile (intermedie)#Esempio|Esempio]]
	- [[#Metodi#anyMatch/allMatch/noneMatch (terminali)|anyMatch/allMatch/noneMatch (terminali)]]
		- [[#anyMatch/allMatch/noneMatch (terminali)#Esempio|Esempio]]
	- [[#Metodi#findFirst/findAny (terminali)|findFirst/findAny (terminali)]]
		- [[#findFirst/findAny (terminali)#Esempio|Esempio]]
	- [[#Metodi#mapToInt e IntStream.summaryStatistics|mapToInt e IntStream.summaryStatistics]]
		- [[#mapToInt e IntStream.summaryStatistics#Esempio|Esempio]]
	- [[#Metodi#flatMap|flatMap]]
		- [[#flatMap#Esempio|Esempio]]
- [[#Collectors|Collectors]]
	- [[#Collectors#Riduzione a singolo elemento|Riduzione a singolo elemento]]
	- [[#Collectors#Riduzione a mappa|Riduzione a mappa]]
	- [[#Collectors#Raggruppamento di elementi|Raggruppamento di elementi]]
		- [[#Raggruppamento di elementi#mapping|mapping]]
	- [[#Collectors#Creare il proprio Collector|Creare il proprio Collector]]
	- [[#Collectors#Partizionamento di elementi|Partizionamento di elementi]]
- [[#IntStream, DoubleStream e LongStream|IntStream, DoubleStream e LongStream]]
	- [[#IntStream, DoubleStream e LongStream#Esempio|Esempio]]
	- [[#IntStream, DoubleStream e LongStream#Passaggio da Stream a IntStream, LongStream, DoubleStream e viceversa|Passaggio da Stream a IntStream, LongStream, DoubleStream e viceversa]]
- [[#Ottenere uno stream infinito|Ottenere uno stream infinito]]
- [[#L’ordine delle operazioni conta|L’ordine delle operazioni conta]]
- [[#Fare copie di stream|Fare copie di stream]]
- [[#Stream paralleli|Stream paralleli]]
	- [[#Stream paralleli#Esempio|Esempio]]
	- [[#Stream paralleli#Quando usare uno stream parallelo?|Quando usare uno stream parallelo?]]
- [[#Le mappe non supportano gli stream|Le mappe non supportano gli stream]]

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

### distinct (intermedia)
Restituisce un nuovo stream senza ripetizione di elementi (gli elementi sono tutti distinti tra loro) basandosi sul metodo equals

#### Esempio
```java
List<Integer> l = List.of(3, 4, 5, 3, 4, 1);
List<Integer> distinti = l.stream()
						  .map(x -> x*x)
						  .distinct()
						  .collect(Collectors.toList());
```

### reduce (terminale)
`reduce` è un’operazione terminale che effettua una riduzione sugli elementi dello stream utilizzando la funzione data in input

#### Esempio
![[Screenshot 2024-05-16 alle 12.48.12.png|450]]
```java
// iterativamente
int somma = 0;
for (int k : lista)
	somma += k;


// tramite stream
lista.stream().reduce(0, (a, b) -> a+b);
// oppure
lista.stream().reduce(0, Integer::sum);
```

Esiste anche una versione di reduce con un solo parametro (senza elemento identità), che restituisce un `Optional<T>`
```java
lista.stream().reduce(Integer::sum);
```

Riduciamo uno stream di `String` a una stringa costruendola elemento per elemento
```java
Optional<String> reduced = l.stream()
							.sorted()
							.reduce((s1, s2) -> s1+"#"+s2);

reduced.ifPresent(System.out::println); // "ab#ac#bb#da"
```

Calcoliamo il prodotto tra interi in una lista:
```java
List<Integer> numbers = List.of(1, 2, 3, 4, 5, 6, 7, 8);
int product = numbers.stream().reduce(1, (a, b) -> a * b);
// product == 40320
```

Calcoliamo il massimo tra gli interi di una lista:
```java
int max = numbers.stream()
				 .reduce(Integer.MIN_VALUE, Integer::max);
// max == 8
```

Calcolo della somma dei valori iva inclusa:
```java
List<Integer> ivaEsclusa = Arrays.asList(10, 20, 30);

// java 7:
double totIvaInclusa = 0.0;
for (int p : ivaEsclusa) {
	double pIvaInclusa = p*1.22;
	totIvaInclusa += pIvaInclusa;
}

// java 8:
ivaEsclusa.stream()
		  .map(p -> p*1.22)
		  .reduce((sum, p) -> sum+p)
		  .orElse(0);
```

### limit (intermedia)
Limita lo stream a `k` elementi (`k` è un `long` passato in input)

#### Esempio
```java
List<String> elementi = List.of("uno", "due", "tre");
List<String> reduced = l.stream()
						.limit(2)
						.collect(toList()); // ["uno", "due"]
```

### skip (intermedia)
Salta `k` elementi (`k` è un `long` passato in input)

#### Esempio
```java
List<String> elementi = List.of("uno", "due", "tre");
List<String> reduced = l.stream()
						.skip(2)
						.collect(toList()); // ["tre"]
```

### takeWhile/dropWhile (intermedie)
`takeWhile` prende elementi finché si verifica la condizione del predicato
`dropWhile` salta gli elementi finché si verifica la condizione

#### Esempio
```java
List<Integer> elementi = List.of(2, 5, 10, 42, 3, 2, 10)
List<Integer> reduced = l.stream()
						 .takeWhile(x -> x<42)
						 .collect(toList());   // [2, 5, 10]
```

### anyMatch/allMatch/noneMatch (terminali)
Gli stream espongono diverse operazioni terminali di matching e restituiscono un booleano relativo all’esito del matching

#### Esempio
```java
boolean anyStartsWithA = l.stream().anyMatch(s -> s.startsWith("a"));
System.out.println(anyStartsWithA); // true
boolean allStartsWithA = l.stream().allMatch(s -> s.startsWith("a"));
System.out.println(allStartsWithA); // false
boolean noneStartsWithZ = l.stream().noneMatch(s -> s.startsWith("z"));
System.out.println(noneStartsWithZ); // true
```

### findFirst/findAny (terminali)
Gli stream espongono due operazioni terminali per ottenere il primo (`findFirst`) o un qualsiasi elemento (`findAny`) dello stream

#### Esempio
```java
List<String> l2 = Arrays.asList("c", "b", "a");
Optional<String> v = l2.stream().sorted().findFirst(); // "a"
```

### mapToInt e IntStream.summaryStatistics
E’ possibile convertire uno `Stream` in un **`IntStream`** attraverso il metodo `mapToInt`. A sua volta lo `IntStream` possiede il metodo `summaryStatistics` che restituisce un oggetto di tipo `IntSummaryStatistics` con informazioni su: minimo, massimo, media, conteggio

#### Esempio
```java
List<Integer> p = List.of(2, 3, 4, 5, 6, 7);
IntSummaryStatistics stats = p.stream()
							  .mapToInt(x -> x)
							  .summaryStatistics();

System.out.println(stats.getMin());     // 2
System.out.println(stats.getMax());     // 7
System.out.println(stats.getAverage()); // 4.5
System.out.println(stats.getCount());   // 6
```

### flatMap
`flatMap` mi è utile nel caso in cui sto lavorando su collezione di collezioni. Questo metodo mi restituisce uno stream “appiattito” con tutti gli elementi della collezione

#### Esempio
```java
// restituisce uno Stream<String[]>
words.map(w -> w.split(""))
	 // restituisce uno Stream<Stream<String>>
	 .map(Arrays::stream);

// con flatMap
Map<String, Long> letterToCount = words.map(w -> w.split("")) // String[]
									   .flatMap(Arrays::stream)
									   .collect(
										   groupingBy(identity(),
										   counting())
										);
```

Supponiamo di avere una lista di stringhe:
```java
// mappa a uno stream di IntStream (quindi uno stream di stream)
l.stream.map(String::chars)
// per risolvere il problema di avere uno stream di stream, posso
// utilizzare flatMap (e, in particolare, essendo un IntStream,
// a flatMapToInt)

l.stream().flatMapToInt(String::chars) // mappa a un unico IntStream
```

Stampare i token (distinti) da file:
```java
Files.lines(Paths.get("stuff.txt"))
	 .map(line -> line.split("\\s+")) // Stream<String[]>
	 .flatMap(Arrays::stream) // Stream<String>
	 .distinct() // Stream<String>
	 .forEach(System.out::println);
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
								  .collect(Collectors.toMap(
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
Come idea `partitionBy(predicato)` risulta essere simile a `groupingBy` infatti questo metodo ci permette di creare una `Map<Boolean, List<T>>` che soddisfano il criterio del predicato in input
```java
Map<Boolean, List<Integer>> m = l.stream()
								 .collect(
								     Collectors.partitioningBy(x -> x%2==0)
								 );
```

---
## IntStream, DoubleStream e LongStream
Si ottengono da uno `Stream` con i metodi `mapToInt`, `mapToLong`, `mapToDouble`. Analoghi metodi sono disponibili nelle 3 classi.

Essi dispongono di 2 metodi statici:
- `range(inizio, fine)` → intervallo esclusivo (aperto a destra)
- `rangeClosed(inizio, fine)` → intervallo inclusivo (chiuso a destra)

### Esempio
Stampa i numeri dispari fino a 10:
```java
IntStream.range(0,10).filter(n -> n%2==1)
					 .forEach(System.out::println);
```

IntStream ottenuto da uno stream di array di interi:
```java
Arrays.stream(new int[] {1, 2, 3}) // restituisce un IntStream
	  .map(n -> 2*n+1)
	  .average() // metodo di IntStream
	  .ifPresent(System.out::println); // 5.0
```

### Passaggio da Stream a IntStream, LongStream, DoubleStream e viceversa
Mediante i metodi `Stream.mapToInt|Double|Long` e, da uno stream di primitivi incapsulati, boxed o `mapToObj`
Tra stream di primitivi, `asDouble|LongStream`. Il metodo `boxed()` in particolare mi permette di tornare da una specializzazione di stream al generico dello stream (es. da `IntStream` a `Stream<Integer>`)

```java
List<String> s = Arrays.asList("a", "b", "c");
s.stream()                  // Stream<String>
 .mapToInt(String::length)  // IntStream
 .asLongStream()            // LongStream
 .mapToDouble(x -> x/42.0)  // DoubleStream
 .boxed()                   // Stream<Double>
 .mapToLong(x -> 1L)        // LongStream
 .mapToObj(x -> "")         // Stream<String>
```

---
## Ottenere uno stream infinito
L'interfaccia `Stream` espone un metodo **`iterate`** che, partendo dal primo argomento, restituisce uno stream infinito con valori successivi applicando la funzione passata come secondo argomento:
```java
Stream<Integer> numbers = Stream.iterate(0, n -> n+10);
```

E' possibile limitare uno stream infinito con il metodo `limit`:
```java
// 0, 10, 20, 30, 40
numbers.limit(5).forEach(System.out::println);
```

Da Java 9, l'interfaccia `Stream` espone un secondo metodo `iterate` che, partendo dal primo argomento, restituisce uno stream di valori successivi applicando la funzione passata come terzo argomento e uscendo quando il predicato del secondo argomento è `false`:
```java
// 0, 10, 20, 30, 40
Stream<Integer> numbers = Stream.iterate(0, n -> n < 50, n -> n+10);
```

---
## L’ordine delle operazioni conta
```java
Stream.of("d2", "a2", "b1", "b3", "c")
	  .map(s -> {
	      System.out.println("map: " + s);
		  return s.toUpperCase();
	  })
	  .filter(s -> {
		  System.out.println("filter: " + s);
		  return s.startsWith("A");
	  })
	  .forEach(s -> System.out.println("forEach: " + s));

// map:     d2
// filter:  D2
// map:     a2
// filter:  A2
// map:     b1
// filter:  B1
// map:     b3
// filter:  B3
// map:     c
// filter:  C
// forEach: A2
```

Invertendo l’ordine di `filter` e `map`  la pipeline è molto più veloce, infatti eseguirà il `filter` su ogni elemento ma il `map` solamente sull’elemento che inizia per “a”
```java
Stream.of("d2", "a2", "b1", "b3", "c")
	  .filter(s -> {
		  System.out.println("filter: " + s);
		  return s.startsWith("a");
	  })
	  .map(s -> {
		  System.out.println("map: " + s);
		  return s.toUpperCase();
	  })
	  .forEach(s -> System.out.println("forEach: " + s));

// filter:  d2
// filter:  a2
// map:     a2
// filter:  b1
// filter:  b3
// filter:  c
// forEach: A2
```

Aggiungiamo un ordinamento dei valori nello stream:
```java
Stream.of("d2", "a2", "b1", "b3", "c")
	  .sorted((s1, s2) -> {
	      System.out.printf("sort: %s, %s\n", s1, s2);
	      return s1.compareTo(s2)
	  })
	  .filter(s -> {
		  System.out.println("filter: " + s);
		  return s.startsWith("a");
	  })
	  .map(s -> {
		  System.out.println("map: " + s);
		  return s.toUpperCase();
	  })
	  .forEach(s -> System.out.println("forEach: " + s));

// sort:    a2; d2
// sort:    b1; a2
// sort:    b1; d2
// sort:    b1; a2
// sort:    b3; b1
// sort:    b3; d2
// sort:    c; b3
// sort:    c; d2
// filter:  a2
// map:     a2
// filter:  b1
// filter:  b3
// filter:  c
// filter:  d2
// forEach: A2
```

Modifichiamo l'ordine delle operazioni e ottimizziamo (in questo caso il `sort` non verrà mai stampato in quanto dopo il `filter` il metodo `sort` non ha elementi da poter confrontare tra loro):
```java
Stream.of("d2", "a2", "b1", "b3", "c")
	  .filter(s -> {
		  System.out.println("filter: " + s);
		  return s.startsWith("a");
	  })
	  .sorted((s1, s2) -> {
	      System.out.printf("sort: %s, %s\n", s1, s2);
	      return s1.compareTo(s2)
	  })
	  .map(s -> {
		  System.out.println("map: " + s);
		  return s.toUpperCase();
	  })
	  .forEach(s -> System.out.println("forEach: " + s));

// filter:  d2
// filter:  a2
// filter:  b1
// filter:  b3
// filter:  c
// map:     a2
// forEach: A2
```

Finalmente l’ordine di esecuzione delle operazioni è ottimizzato:
```java
Stream.of("domus", "aurea", "bologna", "burro", "charro")
	  .map(s -> {
		  System.out.println("map: " + s);
		  return s.toUpperCase();
	  })
	  .anyMatch(s -> {
		  System.out.println("anyMatch: " + s);
		  return s.startsWith("A");
	  });

// map:      domus
// anyMatch: DOMUS
// map:      aurea
// anyMatch: AUREA
```

Nonostante si potrebbe pensare che l’esecuzione delle operazioni in uno stream sia sequenziale, di norma se si associa un metodo `filter` o `map` ad un `limit`, le due operazioni sugli elementi vengono eseguite non su tutti gli elementi dello stream, bensì fin quando non ho raggiunto il numero necessario di elementi per il limit:
```java
// in questo caso non esegue prima tutti i filter e tutti i map
// su tutti gli elementi, bensì viene fatto fino a quando non ho due
// elementi validi
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8);
List<Integer> twoEvenSquares =
			  numbers.stream()
					 .filter(n -> {
					     System.out.println("filtering " + n);
						 return n % 2 == 0;
					 })
					 .map(n -> {
						 System.out.println("mapping " + n);
						 return n * n;
					 })
					 .limit(2)
					 .collect(toList());

// filtering 1
// filtering 2
// mapping   2
// filtering 3
// filtering 4
// mapping   4
```

---
## Fare copie di stream
Nonostante gli stream non siano riutilizzabili, è possibile creare un builder di stream mediante una lambda 

```java
Supplier<Stream<String>> streamSupplier = () ->
	Stream.of("d2", "a2", "b1", "b3", "c")
		  .filter(s -> s.startsWith("a"));

streamSupplier.get().anyMatch(s -> true); // ok
streamSupplier.get().noneMatch(s -> true); // ok
```

Ovviamente ha più senso se, invece di un `Supplier`, abbiamo una funzione che prende in input una `Collection` e restituisce uno stream su tale collection

---
## Stream paralleli
Le operazioni su **stream sequenziali** sono effettuate in un **singolo thread**; le operazioni su **stream paralleli**, invece, sono effettuate concorrentemente su **thread multipli**.

>[!hint]
>`heap` e `metaspace` sono condivisi dai vari thread mentre lo stack è unico per ogni thread

### Esempio
```java
int max = 1000000;
List<String> values = new ArrayList<>(max);
for (int i = 0; i < max; i++) {
	UUID uuid = UUID.randomUUID();
	values.add(uuid.toString());
}
```

Con l’ordinamento sequenziale:
```java
long t0 = System.nanoTime();
long count = values.stream().sorted().count();
System.out.println(count);
long t1 = System.nanoTime();
long millis = TimeUnit.NANOSECONDS.toMillis(t1 - t0);
System.out.printf("ordinamento sequenziale: %d ms", millis);
// ordinamento sequenziale: 899 ms
```

Con l’ordinamento parallelo:
```java
long t0 = System.nanoTime();
long count = values.parallelStream().sorted().count();
System.out.println(count);
long t1 = System.nanoTime();
long millis = TimeUnit.NANOSECONDS.toMillis(t1 - t0);
System.out.printf("ordinamento parallelo: %d ms", millis);
// ordinamento parallelo: 472 ms
```

Tuttavia uno stream parallelo può essere talvolta molto più lento nel caso in cui si tratti di operazioni bloccanti:
```java
List<String> l = IntStream.range(0, 100000)
						  .mapToObj(Integer::toString)
						  .collect(toList());

l.parallelStream/stream()
 .filter(s -> {
	 System.out.format("filter: %s [%s]\n", s, Thread.currentThread().getName());
	 return true;
 })
 .map(s -> {
	 System.out.format("map: %s [%s]\n", s, Thread.currentThread().getName());
	 return s.toUpperCase();
 })
 .sorted((s1, s2) -> {
	 System.out.format("sort: %s <> %s [%s]\n", s1, s2, Thread.currentThread().getName());
	 return s1.compareTo(s2); })
 .forEach(s ->
	 System.out.format("forEach: %s [%s]\n", s, Thread.currentThread().getName())
 );

// parallelStream: 87474402 ns
// stream: 56447233 ns
```

### Quando usare uno stream parallelo?
- Quando il problema è parallelizzabile
- Quando posso permettermi di usare più risorse (es. tutti i core del processore)
- La dimensione del problema è tale da giustificare il sovraccarico (overhead, sovraccarico che devo pagare per la parallelizzazione) dovuto alla parallelizzazione

---
## Le mappe non supportano gli stream
In Java le mappe non supportano gli stream ma forniscono numerose operazioni aggiuntive a partire da Java 8

```java
Map<Integer, String> map = new HashMap<>();
for (int i = 0; i < 10; i++) map.putIfAbsent(i, "val" + i);
map.forEach((id, val) -> System.out.println(val));
```

In Java 9, abbiamo `Map.of` per creare una mappa costante

Se l’elemento 3 è presente, modifica il valore associatogli utilizzando la `BiFunction` in input come secondo parametro:
```java
// se è presente la chiave 3 modificala con val+key
map.computeIfPresent(3, (key, val) -> val + key);
map.get(3); // val33

// se è presente la chiave 9 modifica il suo valore a null
map.computeIfPresent(9, (key, val) -> null);
map.containsKey(9); // false

// se NON è presente la chiave 23, aggiungi l'entry
map.computeIfAbsent(23, key -> "val" + key);
map.containsKey(23); // true

// se NON è presente la chiave 3, associagli "bam",
// altrimenti non fare nulla
map.computeIfAbsent(3, key -> "bam");
map.get(3); // val33

// rimuovi la chiave 3 se associata a "val3"
map.remove(3, "val3");
map.get(3); // val33

// rimuovi la chiave 3 se associata a "val33"
map.remove(3, "val33"); // removes the pair map.get(3);
map.getOrDefault(42, "not found"); // not found

// se è presente la chiave 9, prendi il valore precedente
// e aggiungici "val9"
map.merge(9, "val9", (value, newValue) -> value.concat(newValue));
map.get(9); // val9

// se è presente la chiave 9, prendi il valore precedente
// e aggiungici "concat"
map.merge(9, "concat", (value, newValue) ->
value.concat(newValue));
map.get(9); // val9concat
```