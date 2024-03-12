---
Created: 2024-03-12
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
Le classi vengono inserite (categorizzate) in collezioni dette package e ogni package racchiude classi con funzionalità correlate.
Quando si utilizza una classe è necessario specificarne il package (come per Scanner, che appartiene al package `java.util`). Le classe che abbiamo utilizzato finora (es. System, String) appartengono al package speciale `java.lang` (questo package non deve essere specificato)

Le **API** (Application Programming Interface) di Java sono organizzate in numerosi package
![[Screenshot 2024-03-12 alle 15.12.44.png]]

I package sono rappresentati fisicamente da cartelle (`String.class` si trova sotto `java/lang/`) e una classe può essere inserita in un determinato package semplicemente specificando all’inizio del file (parola chiave package) e posizionando il file nella corretta sottocartella

---
## Importare package
Per evitare di specificare il package d una classe ogni volta. che viene usata, è sufficiente importare la classe o l’intero package (ma attenzione non è ricorsivo)

```java
import java.util.Scanner;
// import java.util.*;

public class ChatBotNonCosiInterattivo {
	public static void main(String[] args) {
		// crea uno Scanner per ottenere l'input da console
		java.util.Scanner input = new Scanner(System.in);
		
		System.out.println("Come ti chiami?");
		
		// legge i caratteri digitati finche' non viene inserito
		// il carattere di nuova riga (l'utente preme invio)
		String nome = input.nextLine();
		System.out.println("Ciao "+nome+"!");
	}
}
```