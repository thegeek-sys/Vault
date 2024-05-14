---
Created: 2024-04-17
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Comparator
```java
public class Person {
	private String firstName;
	private String lastName;
	
	public Person() { /* ... */ }
	public Person(String firstName, String lastName) {
		this.firstName = firstName;
		this.lastName = lastName;
	}
}


Comparator<Person> comparator = (p1, p2) -> p1.firstName.compareTo(p2.firstName);
Person p1 = new Person("Jhon", "Doe");
Person p2 = new Person("Alice", "Wonderland");

// confronto inverso
comparator.reversed().compare(p1, p2); // < 0

// confronto per una chiave specificata
comparator = Comparator.comparing(p -> p.getFirstName());

// confronto per criteri multipli a cascata
comparator = Comparator.comparing(p -> p.getFirstName()
								 .thenComparing(p -> p.getLastName));
comparator = Comparator.comparing(Person::getFirstName)
					   .thenComparing(Person::getLastName);
```

---
## Predicate\<T>
Funzione a valori booleani a un solo argomento generico T

| Modifier and Type         | Method                                                                                                                                             |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `default Predicate<T>`    | **`and(Predicate<? super T> other)`**<br>Returns a composed predicate that represents a short-circuiting logical AND of this predicate and another |
| `static <T> Predicate<T>` | **`isEqual(Object targerRef)`**<br>Returns a predicate that tests if two arguments are equal according to `Object.equals(Object, Object)`          |
| `default Predicate<T>`    | **`negate()`**<br>Returns a predicate that represents the logical negation of this predicate                                                       |
| `default Predicate<T>`    | **`or(Predicate<? super T> other)`**<br>Returns a composed predicate that represents a short-circuiting logical OR of this predicate and another   |
| `boolean`                 | **`test(T t)`**<br>Evaluates this predicate on the given argument                                                                                  |

```java
Predicate<String> predicate = s -> s.length() > 0;
predicate.test("foo"); // true
predicate.test(""); // false

Predicate<String> predicate2 = s -> s.startsWith("f");
predicate.and(predicate2).test("foo"); // true

Predicate<Boolean> nonNull = Objects::nonNull;
Predicate<Boolean> isNull = Objects::isNull;
Predicate<String> isEmpty = String::isEmpty;
Predicate<String> isNotEmpty = String::isEmpty.negate();
```

---
## Function<T,R>
Funzione a un argomento di tipo T e un tipo di ritorno E entrambi generici

| Modifier and Type           | Method                                                                                                                                                                                   |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `default <V> Function<T,V>` | **`andThen(Function<? supper R,? extends V> after)`**<br>Returns a composed function that first applies this function to its input, and then applies the `after` function to the result. |
| `R`                         | **`apply(T t)`**<br>Applies this function to the given argument                                                                                                                          |
| `default <V> Function<V,R>` | **`compose(Function<? super V,? extends T> before`**<br>Returns a composed function that first applies the `before` function to its input, and then applies this function to the result. |
| `static <T> Function<T,T>`  | **`identify()`**<br>Returns a function that always returns its input arguments                                                                                                           |

```java
Function<String, Integer> toInteger = Integer::valueOf;
Integer k = toInteger.apply("123"); // 123

Function<Integer, String> toString = String::valueOf;
String k = toString.apply(123); // "123"

Function<String, String> backToString = toInteger.andThen(toString);
backToString.apply("123"); // "123"

Function<Integer, Integer> square = k -> k*k;
Integer sqr = square.apply(5); // 25
```

---
## Supplier\<T>
Funzione senza argomenti in input

```java
Supplier<String> stringSupplier = () -> "ciao";

Supplier<Person> personSupplier = Person::new;
personSupplier.get(); // new Person();
```

---
## Consumer\<T>
Funzione con un argomento di tipo generico T e nessun tipo di ritorno

```java
Consumer<Person> greeter1 = p -> System.out.println("Hello "+p.firstName);
greeter1.accept(new Person("Luke", "Skywalker"));

Consumer<Person> greeter2 = System.out::println;
```

---
## Optional<\T>
`java.util.Optional` è un contenitore di un riferimento che potrebbe essere o non essere `null` (in modo tale che un metodo può restituire un `Optional` invece di restituire  un riferimento potenzialmente `null`, per  evitare i `NullPointerException`)

### Creare e verificare un Optional
```java
// un optional senza riferimento (contenitore vuoto)
Optional.empty();

// un optional non nullo
Optional.of("bumbumghino");

// un optional di un riferimento che può essere nullo
Optional<String> optional = Optional.ofNullable(s);

// controllo della presenza di un valore non null
Optional.empty().isPresent() == false
Optional.of ("bumbumghigno").isPresent() == true
```

### Ottenere ul valore di un Optional
```java
// mediante orElse
Optional<String> op = Optional.of("eccomi")
op.orElse("fallback"); // "eccomi"
Optional.empty().orElse("fallback"); // "fallback"

// mediante ifPresent o ifPresentOrElse
op.ifPresent(System.out::println); // "eccomi"

// da NON usare
optional.get(); // valore o solleva eccezione se non presente
```