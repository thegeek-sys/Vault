---
Created: 2024-04-10
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction]]
- [[#Sovrascrivere il metodo clone]]

---
## Introduction
L’operatore di assegnazione `=` non effettua una copia dell’oggetto, ma solo del riferimento all’oggetto.
Per creare una copia di un oggetto è necessario richiamare `clone()`. Tuttavia l’implementazione nativa di default di Object.clone copia l’oggetto campo per campo. Per questo non avremo problemi se i campi sono dei tipi primitivi, ma ci troveremo in difficoltà nel caso in cui i campi sono degli oggetti.
Risulta quindi:
- **ottimo** se i campi sono tutti **primitivi** → copierà ogni singolo campo
- **problematico** se i campi sono **oggetti** → in questo caso infatti verrà fatta una copia del campo, ma i campi di questo oggetto (il campo appena clonato) saranno solamente dei riferimenti

---
## Sovrascrivere il metodo clone
Per implementare la copia in una propria classe è necessario sovrascrivere `clone()` che è **protetta** (quindi visibile solo in gerarchia e nel package). Per farlo è necessario implementare l'interfaccia "segnaposto" `Cloneable` altrimenti `Object.clone` emetterà semplicemente l'eccezione `CloneNotSupportedException`

> [!warning]
> Se il nostro oggetto contiene riferimenti e vogliamo evitare che la copia contenga un riferimento allo stesso oggetto membro, non possiamo chiamare semplicemente `super.clone()`

Per etivare la copia dei riferimenti è necessaria la clonazione “profonda” (*deep cloning*)
- Si può utilizzare `Object.clone` per la clonazione dei tipi primitivi
- e richiamare `.clone()` su tutti i campi che sono riferimenti ad altri oggetti, impostando i nuovi riferimenti nell’oggetto clonato

```java
public IntVerctor getCopy() {
	try {
		IntVector v = (IntVector)clone();
		v.list = (ArrayList<Integer>)list.clone();
		return v
	}
}
```