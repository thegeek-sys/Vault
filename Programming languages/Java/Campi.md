---
Created: 2024-03-05
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index
- [[#Introduzione|Introduzione]]
- [[#Struttura|Struttura]]
- [[#Inizializzazione implicita|Inizializzazione implicita]]
- [[#Campi vs. variabili locali|Campi vs. variabili locali]]
- [[#Campi statici]]
- [[#Esempi|Esempi]]
---
## Introduzione
Un campo (detto anche variabile di istanza) costituisce la **memoria privata** di un oggetto (normalmente i campi di una classe sono privati).
Ogni campo ha un **tipo di dati** e un **identificatore** (nome) fornito dal programmatore

---
## Struttura
```java
private [static] [final] tipo_di_dati nome;
```

- `static`
	indica se un campo è **condiviso da tutti**. Definire un campo statico vuol dire che ogni oggetto condividerà la stessa variabile (ci sta solo una locazione di memoria ,se lo modifico in un oggetto verrà modificato in tutti gli oggetti). Un campo di questo tipo è detto *campo di classe*
- `final`
	Indica se il campo è una costante. Questo campo quindi **non può essere modificato** e non può essere riassegnato

> [!warning]
> Da evitare l’uso di una variabile “di comodo” come campo di una classe

---
## Inizializzazione implicita
Al momento della creazione dell’oggetto i campi di una classe sono inizializzati automaticamente

| Tipo del campo    | Inizializzato implicitamente a |
| ----------------- | ------------------------------ |
| `int`, `long`     | `0`, `0L`                      |
| `float`, `double` | `0.0f`, `0.0`                  |
| `char`            | `'\0'`                         |
| `boolean`         | `false`                        |
| `classe X`        | `null`                         |
> [!warning]
> le inizializzazioni sono automatiche per i campi di classe, ma NON per le variabili locali dei metodi

---
## Campi vs. variabili locali

| Campi                                                                                                                  | Variabili locali                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| I campi sono variabili dell’oggetto                                                                                    | Le variabili locali sono variabili definite all’interno di un metodo                                                                                                            |
| sono variabili almeno all’interno di tutti gli oggetti della stessa classe ed esistono per tutta la vita di un oggetto | come parametri del metodo o all’interno del corpo del metodo ed esistono dal momento in cui sono definite fino al termine dell’esecuzione della chiamata al metodo in questione |

---
## Campi statici
Dichiarare un campo statico mi permette di trasformare una variabile di istanza in un **campo di classe**, gli posso dunque accedere da un’altra classe tramite `NomeClasse.nomeCampo`.

Un esempio ne sono le costanti `PI` ed `E` della classe `Math`; queste sono dichiarate:
- `public` → perché accessibili da tutti
- `final` → perché costanti
- `static` → perché non variano secondo lo stato dell’oggetto

`import static` permette di importare campi statici come se fossero definiti nella classe in cui si importano. E’ possibile anche importare tutti i campi statici di una classe `import static java.lang.Math.*`

```java
//import static java.lang.Math.E;
import static java.lang.Math.*;

public class StaticImport {
	public static void main(String[] args) {
		System.out.println(E);
		System.out.println(PI);
	}
}
```

---
## Esempi

```java
public class Hotel {
	private String nome;
	private int numeroStanze;
	private double superficie;
	private boolean bGuidaMichelin;
}
```

