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
- Telefono → secondo standard

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