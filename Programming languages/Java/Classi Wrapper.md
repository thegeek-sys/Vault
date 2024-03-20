---
Created: 2024-03-20
Programming language: "[[Java]]"
Related:
  - "[[Primitivi]]"
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Membri statici delle classi wrapper|Membri statici delle classi wrapper]]
- [[#Autoboxing e auto-unboxing]]
---
## Introduction
Le classi wrapper permettono di convertire i valori di tipo primitivo in un oggetto e forniscono **metodi di accesso** e visualizzazione dei valori

| Tipo primitivo | Classe    | Argomenti del costruttore |
| -------------- | --------- | ------------------------- |
| **byte**       | Byte      | byte o String             |
| **short**      | Short     | short o String            |
| **int**        | Integer   | int o String              |
| **long**       | Long      | long o String             |
| **float**      | Float     | float, double o String    |
| **double**     | Double    | double o String           |
| **char**       | Character | char                      |
| **boolean**    | Boolean   | boolean o String          |
Confrontavamo i valori interi primitivi mediante gli operatori di confronto `==, !=, <, <=, >, >=` ma poiché questi sono relativi solamente agli indirizzi di memoria, per gli oggetti (come appunto le classi wrapper) dobbiamo usare:
- `equals()` → restituisce true se e solo se l’oggetto in input è un intero di valore uguale al proprio
- `compareTo()` → restituisce `0` se sono uguali, `< 0` se il proprio valore è < di quello in ingresso, `> 0` altrimenti

---
## Membri statici delle classi wrapper
- `Integer.MIN_VALUE`, `Integer.MAX_VALUE`
- `Double.MIN_VALUE`, `Double.MAX_VALUE`
- I metodi `Integer.parseInt()`, `Double.parseDouble()` ecc.
- Il metodo `toString()` fornisce una rappresentazione di tipo stringa per un tipo primitivo
- `Character.isLetter()`, `Character.isDigit()`, `Character.isUpperCase()`, `Character.isLowerCase()`, `Character.toUpperCase()`, ecc.

---
## Autoboxing e auto-unboxing
L’*autoboxing* converte automaticamente un tipo primitivo al suo tipo wrapper associato
```java
Integer k = 3;
Integer[] array = { 5, 3, 7, 8, 9 }
```

L’*auto-unboxing* converte automaticamente da un tipo wrapper all’equivalente tipo primitivo
```java
int j = k;
int n = array[j];
```