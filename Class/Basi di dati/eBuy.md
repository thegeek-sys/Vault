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
## Specifica dei tipi di dato
- Condizione → {ottimo, buono, discreto, da_sistemare}
- URL → secondo standard
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
`popolarità():Stringa`
- precondizioni → sia `p:Post` tale che esista almeno un link `(this, p):pubblica`
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `u:UtentePrivato`
		- sia `a:Asta`
		- sia `A` l’insieme dei link `(u, a):bid` tale che `adesso-bid.istante<=12 mesi`
		- sia `c:CompraSubito`
		- sua `C` l’insieme dei link `(u, c):acquista` tale che `adesso-acquista.istante<=12 mesi
		- `result="bassa"` se `|C|+|A|<50`
		- `result="media"` se `50<=|C|+|A|<=300`
		- `result="media"` se `|C|+|A|>300`

### UtenteRegistrato
Un’istanza di questa classe rappresenta un utente registrato
#### Specifica delle operazioni di classe
`affidabilità():Reale`
- precondizioni → sia `f:Feedback` tale che esista almeno un link `(this, f):pubblica`
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `F` l’insieme