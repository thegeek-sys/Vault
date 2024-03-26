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

| <div style="width:315px;text-align:center">Metodo</div> | Descrizione                                                                                                           |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `Object clone()`                                        | Restituisce una copia dell’oggetto                                                                                    |
| `boolean eqauls(Object o)`                              | Confronta l’oggetto con quello in input                                                                               |
| `Class<? extends Object> getClass()`                    | Restituisce un oggetto di tipo Class che contiene informazioni sul tipo dell’oggetto                                  |
| `int hashCode()`                                        | Restituisce un intero associato all’oggetto (per es. ai fini della memorizzazione in strutture dati, hashtable, ecc.) |
| `String toString()`                                     | Restituisce una rappresentazion                                                                                       |
https://docs.oracle.com/javase/8/docs/api/java/lang/Object.html

> [!hint]
> Class è una classe esiste nel JRE per gestire le relazioni tra i tipi. Se faccio un getClass su `a` definita come `Animale a = new Gatto();` questo mi restituirà la classe Gatto

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

Però esistono due differenti visioni chi predilige l’utilizzo di `instanceof` e chi invece preferisce `getClass()`

> “The reason that I favor the instanceof approach is that, when you use the getClass approach, you have the restriction that objects are only equal to other objects of the same class, the same run time type. If you extend a class and add a couple of innocuous methods to it, then check to see whether some object of the subclass is equal to an object of the super class, even if the objects are equal in all important aspects, you will get the surprising answer that they aren't equal. In fact, this violates a strict interpretation of the Liskov substitution principle, and can lead to very surprising behavior. In Java, it's particularly important because most of the collections (HashMap, etc.) are based on the equals method. If you put a member of the super class in a hash table as the key and then look it up using a subclass instance, you won't find it, because they are not equal.”

\- Joshua Bloch
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
		if (getClass() != o.getClass()) return false;
		Punto p = Punto(o);
		return x==p.x && y==p.y && z==p.z;
	}
}
```