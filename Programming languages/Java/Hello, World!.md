---
Created: 2024-02-29
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index

1. [[#Introduzione|Introduzione]]
2. [[#Argomenti in input|Argomenti in input]]
---

![[Screen Shot 2024-02-28 at 11.13.41 1.png]]
## Introduzione
Un programma java si salva in un file unicode, con il titolo dell'identificatore della dichiarazione della classe.

```java
'''
HelloWorld.java
'''

public class HelloWorld
// corpo della classe
{
	public static void main(String[] args)
	// corpo del metodo main
	{
		System.out.print("Hello, World!");
		System.out.println();
	}
}
```
Un programma deve quindi iniziare con una dichiarazione di una classe (il cui titolo sarà il nome del file), seguito da un metodo chiamato `public static void main` che riceve un array stringhe.

---
## Argomenti in input
Gli args possono essere passati come argomenti in entrata - `Strings[]` rappresenta l'array di stringhe fornite sulla command line dopo il nome del file.

```java
public class BotSempliceSemplice {
	public static void main (String[] args) {
		System.out.print("Ciao");
		System.out.print(args[0]); // prima parola presa in input
		System.out.println(". Come va?");
	}
}

- Compila: javac BotSempliceSemplice.java
- Esegui: java BotSempliceSemplice Pippo
- Output: Ciao Pippo. Come va?
```
