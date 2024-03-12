---
Created: 2024-03-12
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## java.util.Scanner
Per leggere in input da console, non come args, utilizzo la classe `java.util.Scanner` costruita passando al costruttore lo strem di input (`System.in` di tipo `java.io.InputStream`).

```java
public class ChatBotNonCosiInterattivo {
	public static void main(String[] args) {
		// crea uno Scanner per ottenere l'input da console
		java.util.Scanner input = new java.util.Scanner(System.in);
		
		System.out.println("Come ti chiami?");
		
		// legge i caratteri digitati finche' non viene inserito
		// il carattere di nuova riga (l'utente preme invio)
		String nome = input.nextLine();
		System.out.println("Ciao "+nome+"!");
	}
}
```

---
