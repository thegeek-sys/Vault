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

## Predicate\<T>
Funzione a valori booleani a un solo argomento generico T

| Modifier and Type         | Method                                                                                                                                             |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `default Predicate<T>`    | **`and(Predicate<? super T> other)`**<br>Returns a composed predicate that represents a short-circuiting logical AND of this predicate and another |
| `static <T> Predicate<T>` | **`isEqual(Object targerRef)`**<br>Returns a predicate that tests if two arguments are equal according to `Object.equals(Object, Object)`          |
| `default Predicate<T>`    | **"`negate()`**<br>Returns a predicate that represents the logical negation of this predicate                                                      |
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

