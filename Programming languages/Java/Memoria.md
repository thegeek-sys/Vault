---
Created: 2024-03-07
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index

- [[#Introduzione|Introduzione]]
- [[#Static|Static]]
- [[#Istruzioni all’avvio del compilatore|Istruzioni all’avvio del compilatore]]
- [[#Rappresentazione memoria|Rappresentazione memoria]]
	- [[#Rappresentazione memoria#Esempio|Esempio]]

---
## Introduzione
Esistono due tipi di memoria:
- **heap**: Vengono allocate le aree di memoria per la _creazione dinamica_ (oggetti)
- **stack**: Vengono allocate le variabili locali

![[Screenshot 2024-03-07 alle 13.35.24.png]]![[Screenshot 2024-03-07 alle 13.35.33.png]] ![[Screenshot 2024-03-07 alle 13.35.41.png]]

---
## Static
I campi di una classe possono essere dichiarati static (relativo dell’intera classe).
Un campo static esiste **una sola locazione di memoria**, allocata prima di qualsiasi oggetto della classe in una zona speciale di memoria nativa chiamata *MetaSpace*. Viceversa, per ogni campo non static esiste una locazione di memoria per ogni oggetto, allocata a seguito dell’istruzione new

---
## Istruzioni all’avvio del compilatore
Quando viene avviato un programma il compilatore di Java esegue queste azioni:
- Tramite il **ClassLoader** carica la classe e vede se c'è un metodo main eseguibile.
- Alloca nel **MetaSpace** i campi statici se ci sono
- Inizia ad eseguire il main riservandogli una locazione di memoria nello **stack** e copiando **nell'heap** i valori presenti nella lista args, anche se vuota.
- Va avanti nella compilazione del programma allocando locazioni di memoria di variabili, oggetti e metodi mano a mano che li troviamo.

I valori dei parametri di un metodo vengono sempre copiato e non passati per riferimento, se troviamo un nuovo metodo questo viene allocato ad un livello superiore nello stack.

---
## Rappresentazione memoria

![[Screenshot 2024-03-12 alle 09.16.37.png]]

Quando viene caricata una classe la prima cosa che la JVM è controllare se ci sono campi statici che vengono quindi allocati nel metaspace (questi essendo campi di classe vengono inizializzati implicitamente).
A questo punto viene creato un frame di attivazione nello stack il metodo main che viene chiamato all’avvio di un programma. All’interno di questo frame metterò le variabili locali che vengono allocate. Se ad esempio passo ad un oggetto una variabile questa sarà allocata all’interno dell’heap collegato tramite una freccia all’oggetto stesso

### Esempio

```java
public class Tornello {
	static private int passaggi;
	
	public void passa() {passaggi++;}
	public static void int getPassaggi() {return passaggi;}
	
	public static void main(String[] args) {
		Tornello t1 = new Tornello();
		t1.passa();
		Tornello t2 = new Tornello();
		for (int k=0; k<10; k++) t2.passa();
		int g;
		String s=null;
		// fotografa lo stato della memoria
	}
}
```

![[Screenshot 2024-03-12 alle 09.43.54.png]]

>[!info]
>`null` è una parola chiave utilizzata per una variabile o un campo di tipo oggetto (es. String, vettori) e intende dire che non ci sta alcun riferimento ad un oggetto nell’heap (per questo da s non viene fatta alcuna freccia).


Tutti i passaggi
![[document-92-109.pdf]]


---
## Costo oggetti in memoria
Un oggetto qualsiasi costa minimo 8 byte (informazione di base come la classe dell’oggetto, flag id status, ID ecc.)
$\text{Integer } 8 \text{ byte dell’oggetto } + 4 \text{ byte per l’int } + \text{ padding (spazio che separa di vari oggetti) } = 16 \text{ byte}$
$\text{Long }8+8=16\text{ byte}$

Un riferimento “costerebbe” 8 byte ma si usano i *compressed oop* (ordinary object pointer) che sono object offset da 32 bit (ogni 8 byte)quindi indicizzano fino a 32Gb di RAM (attivi fino a – Xmx32G), quindi richiedono normalmente 4 byte

Un array richiede minimo 12 byte (gli 8 di qualsiasi oggetto più 4 per la length)
$$
\begin{split}
\text{Una stringa (in Java 8)} &= 2\cdot \text{numero di caratteri (codifica Unicode)} + 8 \text{ (suo overhead) } +\\&+ 4 \text{ (reference all'array char[])} + 12 \text{ (char[] array overhead)} + 4 \text{(hash)} \\&= 2\cdot \text{(hash)} + 28
\end{split}
$$
(Da Java 9 le stringhe sono state reimplementate introducendo le stringhe
“compatte” che utilizzano un solo byte se tutti i suoi caratteri usano l’encoding
LATIN-1 (extended ASCII))

**Tutti + arrotondamento a un multiplo di 8**
Per il padding, tutti gli oggetti vengono "allineati" a multipli di 8 (64 bit)