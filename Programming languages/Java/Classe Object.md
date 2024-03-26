---
Created: 2024-03-26
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
Tutte le classi in Java ereditano direttamente o indirettamente dalla classe **Object**, con tutti i relativi 11 metodi

Quando si definisce una classe senza estenderne
un’altra:
```java
public class LaMiaClasse {}
```
questo equivale a estendere `Object`
```java
public class LaMiaClasse extends Object {}
```

---
## Metodi principali

|Metodo|Descrizione|
|---|---|
|`Object clone()`|Restituisce una copia dell’oggetto|
|`boolean eqauls(Object o)`|Confronta l’oggetto con quello in input|
|`Class<? extends Object> getClass()`|Restituisce un oggetto di tipo Class che contiene informazioni sul tipo dell’oggetto|
|`int hashCode()`|Restituisce un intero associato all’oggetto (per es. ai fini della memorizzazione in strutture dati, hashtable, ecc.)|
|`String toString()`|Restituisce una rappresentazion|
https://docs.oracle.com/javase/8/docs/api/java/lang/Object.html

---
## Sovrascrivere il metodo toString
`toString` è uno dei metodi, che non prende argomenti e restituisce una String, che ogni classe eredita direttamente o indirettamente dalla classe Object. 
Questo metodo viene chiamato implicitamente quando un oggetto deve essere convertito a String (es. `System.out.println(o)`)

### Esempio
```java
public class Punto {
	private int x, y, z;
	
	public Punto(int x, int y, int z) {
		this.x = x;
		this.y = y;
		this.z = z;
	}
	
	@Override
	public String toString() {
		return "("+x+", "+y+", "+z+")";
	}
}
```

---
## Sovrascrivere il metodo equals
Il metodo equals viene invocato per confrontare il contenuto di due oggetti. Per default, se sono “uguali”, il metodo restituisce true.

Tuttavia, la classe **Object** non conosce il contenuto delle sottoclassi, infatti per mantenere il “contratto” è necessario sovrascriverlo

### Esempio
```java
public class Punto {
	private int x, y, z;
	
	public Punto(int x, int y, int z) {
		this.x = x;
		this.y = y;
		this.z = z;
	}
	
	@Override
	public boolean equals(Object o) {
		if (o == null) return false;
		if (getClass() != o.getClass)
	}
}
```