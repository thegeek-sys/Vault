---
Created: 2024-04-17
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
In Java è possibile passare **riferimenti a metodi esistenti** utilizzano la sintassi:
- `Classe::metodoStatico`
- `riferimentoOggetto::metodoNonStatico`
- `Classe::metodoNonStatico`

### Esempio
```java
Converter<String, Integer> converter = Integer::valueOf;
Integer converted = converter.convert("123")
```

---
## Riferimento usando il nome della classe vs. riferimento a un oggetto
Che differenza ci sta tra `riferimentoOggetto::metodo` e `nomeClasse::metodo`?
- Nel primo caso, il metodo sarà chiamato **sull’oggetto riferito**.
- Nel secondo caso, **non stiamo specificando su quale oggetto applicare il metodo**.
	Ma il metodo è d’istanza, quindi utilizza i membri d’istanza (campi, metodi, ecc.).
	Ci si riferisce al metodo **implicitamente esteso** con un primo parametro aggiuntivo: un riferimento a un oggetto della classe cui appartiene il metodo

### Esempio: riferimento a metodi d’istanza mediante classe
Considerate `Arrays.sort(T[] a, Comparator<? super T> c)` e `Comparator`
```java
@FunctionalInterface
public interface Comparator<T> {
	int compare(T o1, T o2);
	boolean equals(Object o);
}

// Posso scrivere
Arrays.sort(new String[] {"a", "c", "b"}, String::compareTo)
```

Oppure:
```java
@FunctionalInterface
public interface StringProcessor {
	String process(String s);
}

// riferimento a metodo d'istanza String toLowerCase()
StringProcessor f = String::toLowerCase;
System.out.println(f.process("BELLA")); // "bella"

String s = "bella";
// riferimento a metodo d'istanza String concat(String s)
StringProcessor g = s::concat;
System.out.println(g.process("zi!")); // "bella zi!"

// riferimento a metodo statico String valueOf(Object o)
StringProcessor h = String::valueOf; 
System.out.println(h.process("bella")); // "bella"
```

