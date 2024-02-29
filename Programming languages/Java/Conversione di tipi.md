---
Created: 2024-02-29
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduzione
In Java posso utilizzare posso utilizzare dei metodi (impliciti o espliciti) che mi permettono di fare casting o conversione tra tipi diversi.

> [!info] 
> Il casting ha precedenza più elevata rispetto agli operatori aritmetici

---
## Conversione esplicita
Nella conversione esplicita utilizziamo un metodo che prende in ingresso un argomento di un tipo e restituisce un valore di altro tipo.
Tra questi metodi troviamo:
- `Integer.parseInt()`
- `Double.parseDouble()`
- `Math.round()`
- `Math.floor()`
- `Math.ceil()`

Caso d’uso
```java
public class SommaInteri {
	public static void main(String[] args) {
		int a = Integer.parseInt(args[0])
		int b = Integer.parseInt(args[0])
		System.out.println(a+b)
	}
}
```

---
## Cast esplicito
Il casting esplicito viene utilizzato quando si vuole passare da un tipo **più preciso ad un tipo meno preciso** (solo numerici). Questo non fa rounding, ma solamente un troncamento (es. da double ad int viene presa solamente la parte intera).

```java
double v = 2.746327;
int conv = (int)v; // 2
```

---
## Cast implicito
Se il tipo di partenza è meno preciso (es. int → double) Java può automaticamente convertire il valore al tipo più preciso.
Inoltre quando un `int` viene sommato ad una stringa il tipo in output è `String`
```java
int i = 5;
double d = 4.5;
String s = "La somma fa: ";
String f = s+d+i;
System.out.println(f)
```

In particolar modo in **assegnazione**:
- `byte`, `short` e `char` possono essere promossi a **`int`**
- `int` può essere promosso a `long`
- `float` può essere promosso a `double`
Nella fase di **calcolo di un’espressione**:
- se uno dei due  operandi è `double`, l’intera espressione è promossa a `double`
- altrimenti se uno d egli operandi è `float`, l’intera espressione sarà promossa a `float`

```java
double d = 2; // 2.0
```

