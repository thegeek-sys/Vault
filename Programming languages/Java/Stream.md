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

---
## Ottenere un o stream
Possiamo ottenere uno stream direttamente dai dati con il metodo statico generico `Stream.of(array di un certo tipo)`.
Da Java 8 l'interfaccia `Collection` è stata estesa per includere due nuovi metodi di default:
- **`default Stream<E> stream()`** → restituisce uno stream sequenziali
- **`default Stream<E> parallelStream()`** → restituisce un nuovo stream parallelo, se possibile (altrimenti restituisce uno stream sequenziale)

### Esempio
E' possibile ottenere uno stream anche per un array, con il metodo statico `Stream<T> Arrays.stream(T[ ] array)`
E' possibile ottenere uno stream di righe di testo da `BufferedReader.lines()` oppure da `Files.lines(Path)`
E’ possibile ottenere uno stream di righe anche da `String.lines