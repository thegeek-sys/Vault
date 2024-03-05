---
Created: 2024-02-29
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index

- [[#Introduzione|Introduzione]]
- [[#Conversione esplicita|Conversione esplicita]]
- [[#Cast esplicito|Cast esplicito]]
- [[#Cast implicito|Cast implicito]]
- [[#Tabella dei tipi|Tabella dei tipi]]
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
- `Integer.toString()`
- `Math.round()`
- `Math.floor()`
- `Math.ceil()`

Caso d’uso
```java
public class SommaInteri {
	public static void main(String[] args) {
		int a = Integer.parseInt(args[0])
		double b = Double.parseDouble(args[0])
		
		String sA = Integer.toString(a)
		String sB = Double.toString(b)
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
String v = 'ciao' + (5+3) // "ciao8"
String s = 'ciao' + 5 + 3 // "ciao53"
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

int i = 5;
double d1 = 4.5;
System.out.println(i+d1) // 9.5
```

---
## Tabella dei tipi

| Espressione                | Tipo   | Valore |
| -------------------------- | ------ | ------ |
| `(int)2.71828`             | int    | 2      |
| `Math.round(2.71828)`      | long   | 2      |
| `(int)Math.round(2.71828)` | int    | 2      |
| `(int)Math.round(3.14159)` | int    | 3      |
| `Integer.parseInt("42")`   | int    | 42     |
| `"42" + 99`                | String | “4299” |
| `42 * 0.4`                 | double | 16.8   |
| `(int)42 * 0.4`            | double | 16.8   |
| `42 * (int)0.4`            | int    | 0      |
| `(int)(42 * 0.4)`          | int    | 16     |
