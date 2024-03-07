---
Created: 2024-03-07
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Metodi|Metodi]]
	- [[#`length()`|length()]]
	- [[#`toUpperCase()`, `toLowerCase()`|toUpperCase(), toLowerCase()]]
	- [[#`atChar(int)`|atChar(int)]]
	- [[#`substring(int, int)`|substring(int, int)]]
	- [[#`concat(String)`|concat(String)]]
	- [[#`indexOf(String)`|indexOf(String)]]
	- [[#`replace(String)`|replace(String)]]
	- [[#`split(String)`|split(String)]]
---
## Introduzione
La classe `java.lang.String` è una classe fondamentale perché è relativo a un tipo di dato i cui letterali sono parte della sintassi del linguaggio e per il quale il significato dell’operatore + è ridefinito.
Non richiede import perché appartiene al package “speciale” `java.lang`

---
## Metodi
#### `length()`
La classe String, essendo una sequenza di char, è dotata del metodo `length()`
```java
String s = “ciao”;
System.out.println(s.length());
```

---
#### `toUpperCase()`, `toLowerCase()`
Con i metodi `toLowerCase` e `toUpperCase` si ottiene un’altra stringa tutta maiuscola o minuscola. La stringa su cui viene invocato il metodo non viene modificata.
Posso inoltre chiamare un metodo su una stringa direttamente senza dover invocare la Classe per la creazione dell’istanza.
```java
String min = "Ciao".toLowerCase(); // "ciao"
String max = "Ciao".toUpperCase(); // "CIAO"
String ariMin = max.toLowerCase(); // "ciao"
```

---
#### `atChar(int)`
Posso ottenere il k-esimo carattere di una striga con il metodo `charAt` e restituisce un tipo `char` (non `String`).
- il primo carattere è in posizione `0`
- l’ultimo carattere è in posizione `stringa.length()-1`
```java
"ciao".charAt(2) // "a"
```

---
#### `substring(int, int)`
E’ possibile ottenere una sottostringa di una stringa con il metodo `substring(startIndex, endIndex)`. Dove:
- `startIndex` → indica l’indice di partenza della sottostringa
- `endIndex` → indice successivo all’ultimo carattere della sottostringa
Esiste anche la versione `substring(startIndex) `che equivale a `substring(startIndex, stringa.length())`
```java
String s = "ciao";
s.substring(1, 3) // "ia"
```

---
#### `concat(String)`
La concatenazione tra due stringhe può essere effettuata con l’operatore speciale + oppure mediante il metodo `concat(s)`
```java
String s2 = s1+s2
String s4 = s1.concat(s2)
```

Tuttavia, se si devono concatenare parecchie stringhe, è bene utilizzare la classe **`StringBuilder`**, dotata dei metodi `append(String s)` e `insert(int posizione, String s)`
```java
StringBuilder sb = new StringBuilder();
sb.append(s1).append(s2)
String s5 = sb.toString();
```

---
#### `indexOf(String)`
Si può cercare la (prima) posizione di un carattere o di una stringa con `indexOf(c)`. Questo metodo restituisce `-1` se il carattere/stringa non è presente
```java
int k = "happy happy bithday".indexOf('a'); // 1
int j = "din din don don".indexOf("don"); // 8
int h = "abcd".indexOf('e'); // -1
```

---
#### `replace(String)`
Con il metodo `replace` è possibile sostituire le occorrenze di un carattere o di una stringa all’interno di una stringa
```java
String s1 = "uno_due_tre".replace('_')
```

---
#### `split(String)`
Il metodo `split` prende in input un’espressione regolare `s` (senza entrare in dettagli, è sufficiente pensarla come una semplice stringa) e restituisce un array di sottostringhe separate da `s`
```java
String[] parole = "uno due tre".split(" ")
```