---
Created: 2024-03-12
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#if|if]]
- [[#Operatore condizionale|Operatore condizionale]]
- [[#switch|switch]]
- [[#while|while]]
- [[#do while|do while]]
- [[#for|for]]
- [[#Uscire dal ciclo|Uscire dal ciclo]]
	- [[#Uscire dal ciclo#break vs. return|break vs. return]]
- [[#Saltare all’iterazione successiva|Saltare all’iterazione successiva]]

---
## Introduction
In Java possiamo prendere decisioni attraverso istruzioni di controllo **condizionali** (istruzioni che possono essere o non eseguite sulla base di certe condizioni) e **iterative** (istruzioni che devono essere eseguite ripetutamente sulla base di certe condizioni)

---
## if
Per realizzare una decisione si usa l’istruzione `if`. La sintassi è:
```java
if (<espressione booleana>) <singola istruzione>;


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
La sintassi dell’istruzione `while` è simile a quella dell’`if`, la differenza è che istruzioni del corpo sono eseguite finché l’espressione booleana è vera (viene controllata all’inizio di ogni esecuzione del corpo). Appena l’espressione booleana è falsa (eventualmente anche subito), il ciclo termina

```java
while (<espressione booleana>) {
	<istruzioni>;
}
```

---
## do while
Questo costrutto si comporta esattamente come il `while` ma la condizione di uscita viene verificata alla fine dell’esecuzione del corpo del ciclo (invece che all’inizio). Questo permette almeno una prima esecuzione del codice nonostante l’espressione booleana sia in partenza falsa

```java
do {
	<istruzioni>;
} while (<espressione booleana>);
```

---
## for
E’ un costrutto alternativo al `while` che fornisce più flessibilità nella realizzazione di cicli.

```java
for (<inizializzazione>; <espressione booleana>; <incremento>) {
	<istruzioni>;
}


for (<inizializzazione>; <espressione booleana>; <incremento>)
	<istruzione>;



// esempio
int somma = 0;
for (int k = 0; k <= N; k++) {
	somma += k;
	System.out.println(somma);
}
```

 Lo schema è il seguente:
 - **inizializza** la variabile “di controllo”
 - **esegui il test d’uscita** sull’espressione booleana
 - **esegui il corpo del for**
 - alla fine di ogni ciclo **incrementa/decrementa il valore della variabile di controllo** come specificato

All’interno dei cicli for posso anche inizializzare e incrementare più variabili allo stesso tempo
```java
for (int k = 0, i = 0; i <= 10; i++, k+=5) {
	// codice iterazione
}
```

---
## Uscire dal ciclo

Indipendentemente dal tipo di ciclo, può essere necessario uscire dal ciclo durante l’esecuzione del suo corpo. Questo è possibile attraverso l’istruzione `break` (utilizzabile solo all’interno di un ciclo)

> [!info]
> L’istruzione `break` mi permette di uscire solo dal ciclo che lo contiene.
> Se invece voglio uscire da cicli annidati devo utilizzare l’istruzione `break <etichetta>`

Esempio:
```java
outer:
for (int i=0; i<h; i++) {
	for (int j = 0; j<w) {
		// codice qui
		// ...
		if (j == i) break outer;
	}
}

// una volta eseguito il break mi ritrovo qui
```

### break vs. return
Mentre l’istruzione `return` interrompe l’esecuzione del metodo, l’istruzione `break` interrompe l’esecuzione di un ciclo (for, while, do…while)

## Saltare all’iterazione successiva
Può anche essere utile saltare all’iterazione successiva. Questo viene fatto attraverso l’istruzione `continue` usata all’interno del ciclio.
Questo significa che non verranno eseguite le istruzioni successive al continue ma si passerà direttamente alla prossima iterazione.