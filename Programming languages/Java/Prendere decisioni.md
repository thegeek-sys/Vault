---
Created: 2024-03-12
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
In Java possiamo prendere decisioni attraverso istruzioni di controllo **condizionali** (istruzioni che possono essere o non eseguite sulla base di certe condizioni) e **iterative** (istruzioni che devono essere eseguite ripetutamente sulla base di certe condizioni)

---
## if
Per realizzare una decisione si usa l’istruzione `if`. La sintassi è:
```java
if (<espressione booleana>) <singola istruzione>;
```

Oppure:
```java
if (<espressione booleana uno>) {
	<istruzioni caso uno true>;
} else if (<espressione booleana due>) {
	<istruzioni caso due true>;
} else {
	<istruzioni nessuno dei precedenti>;
}
```


E’ inoltre importante ricordare che l’else, quando non vengono utilizzate le graffe, si riferisce sempre all’istruzione if immediatamente precedente. Quindi:
```java
if (x > 0)
	if (y > 0)
		System.out.println("x e y sono > 0")
	else
		System.out.println("y è < di 0")
```

---
## Operatore condizionale
In Java (come in C) esiste un operatore di selezione (operatore condizionale) nella forma di espressione (chiamato in gergo operatore “*elvis*“)

```java
<condizione> ? <caso true> : <caso false>

// esempio
int abs = x < 0 ? -x : x;
String testaCroce = Math.random() < 0.5 ? "Testa" : "Croce";
```

---
## switch
Per confrontare il valore di un’espressione intera o convertibile a intero (0, da Java 7 in poi, un valore stringa), si può utilizzare l’istruzione `switch`

```java
switch (<espressione intera>) {
	case <valore1>: <istruzione>; break;
	case <valore2>: <istruzione>; break;
	...
	case <valoren>: <istruzione>; break;
	
	[default]: <istruzione>; break;
}
```

E’ possibile anche utilizzare una notazione contratta con l’operatore `->` che non richiede il break per uscire (da Java 13 in poi)
```java
switch(k % 7) {
	case 0 -> iniziaLaSettimana();
	case 1 -> faiLaSpesa();
	/* ... */
	case 5 -> relax();
	case 6 -> gitaDellDomenica();
}
```

Questo tipo di notazione ci permette anche di restituire un’espressione
```java
// java >= 13
String s = switch(c) {
	case 'k' -> "kappa";
	case 'h' -> "acca";
	case 'l' -> "elle";
	case 'c' -> "ci";
	case 'a', 'e', 'i', 'o', 'u' -> "vocale "+c;
	default -> non so leggerlo
}

// java < 13
String s;
switch(c) {
	case 'k': s = "kappa"; break;
	case 'h': s = "acca"; break;
	case 'l': s = "elle"; break;
	case 'c': s = "ci"; break;
	case 'a': case: 'e' case: 'i'
	case: 'o' case: 'u':
		s = "vocale "+c; break;
	default: non so leggerlo
}
```

---
## while
La sintassi dell’istruzione `while` è simile a quella dell’`if`

```java
while (<espressione booleana>) {
	<istruzioni>;
}
```