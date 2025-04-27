---
Created: 2025-04-23
Class: "[[Basi di dati]]"
Related:
---
---
## Raffinamento dei requisiti
- Utenti registrati
	- nome (una stringa)
	- data di registrazione (una data)
- gli utenti registrati possono pubblicare dei **post**
	- descrizione (una stringa)
	- categoria a cui appartiene
	- metodi di pagamento accettati (bonifico o carta di credito)
	- se l’oggetto è nuovo o usato
	- i posto possono essere di due tipi
		- asta
			- prezzo iniziale
			- rialzo
			- scadenza dell’asta
		- compralo subito
			- solo il prezzo di vendita

---
## Diagramma UML delle classi
![[Pasted image 20250423210847.png]]

---
## Specifica dei tipi di dato
- Condizione → {ottimo, buono, discreto, da_sistemare}
- URL → secondo standard
- Popolarità → {bassa, media, alta}
---
## Specifica di classe
### AstaConclusa
Un’istanza di questa classe rappresenta un’asta conclusa
#### Specifica delle operazioni di classe
`acquirente():Utente`
- precondizioni → sia `u:Utente` tale che esista almeno un link `(this, u):bid`
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `R` l’insieme dei link `(this, u):bid`
		- `result=u` tale che `bid.istante` sia il massimo tra tutti i link in `R`

`prezzo_vendita():Reale>=0`
- precondizioni → sia `u:Utente` tale che esista almeno un link `(this, u):bid`
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `R` l’insieme dei link `(this, u):bid`
		- `result=|R|*this.rialzo+this.prezzo_iniziale`

### Categoria
Un’istanza di questa classe rappresenta una categoria degli oggetti
#### Specifica delle operazioni di classe
`sottocategorie():Categoria [0..*]`
- precondizioni → nessuna
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `c:Categoria`
		- sia `R` l’insieme dei link `(this, c):gerarchia` tale che `c` abbia il ruolo di sottocategoria
		- sia `S` l’insieme degli oggetti `c` nell’insieme `R`
		- `result=S`

### VenditoreProfessionale
Un’istanza di questa classe rappresenta un venditore professionale
#### Specifica delle operazioni di classe
`popolarità():Popolarità`
- precondizioni → sia `p:Post` tale che esista almeno un link `(this, p):pubblica`
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `u:UtentePrivato`
		- sia `a:Asta`
		- sia `A` l’insieme dei link `(u, a):bid` tale che `adesso-bid.istante<=12 mesi`
		- sia `c:CompraloSubito`
		- sua `C` l’insieme dei link `(u, c):acquista` tale che `adesso-acquista.istante<=12 mesi
		- `result=bassa` se `|C|+|A|<50`
		- `result=media` se `50<=|C|+|A|<=300`
		- `result=alta` se `|C|+|A|>300`

### UtenteRegistrato
Un’istanza di questa classe rappresenta un utente registrato
#### Specifica delle operazioni di classe
`affidabilità():Reale 0..1`
- precondizioni → sia `p:Post` tale che esista almeno un link `(this, p):feedback`
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `p:Post`
		- sia `F` l’insieme delle relazioni `(this, p):feedback`
		- sia `z=|{f t.c. f in F e f.valutazione<=2}|/|F|`
		- sia `u` la somma di tutti gli `f.valutazione` in `F`
		- sia `m=u/|F|`
		- `result=m*(1-z)/5`

### UtentePrivato
Un’istanza di questa classe rappresenta un utente privato

`[UtentePrivato.utente_feedback.verifica_acquisto]`
Per ogni `p:Post`, `u:UtentePrivato` nella relazione `(u,p):feedback` allora:
- se `p:CompraloSubito` allora esiste la relazione `(u,p):acquista`
- se `p:AstaConclusa` allora `p.acquirente()=u`