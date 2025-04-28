---
Created: 2025-04-23
Class: "[[Basi di dati]]"
Related:
---
---
## Raffinamento dei requisiti
1. utenti registrati
	1. nickname
	2. data registrazione
	3. post pubblicati (vedi req. 2)
	4. affidabilità (operazione)
2. post (annunci per la vendita di **singoli** oggetti)
	1. descrizione
	2. categoria a cui appartiene l’oggetto (vedi req. 4)
	3. metodi di pagamento accettati (bonifico o carta di credito)
	4. stato dell’oggetto
		1. nuovo
			1. durata della garanzia (obbligatorio)
		2. usato
			1. durata della garanzia (non obbligatorio)
			2. condizioni (ottimo, buono, discreto, da sistemare)
	5. tipologia di post
		1. con asta al rialzo
			1. prezzo iniziale dell’asta
			2. prezzo dei singoli rialzi (in euro)
			3. istante di scadenza dell’asta
			4. se è stata conclusa
				1. il bid che si è aggiudicato l’oggetto in vendita (se esiste)
		2. “compralo subito”
			1. prezzo di vendita dell’oggetto
3. bid (offerte a post con asta al rialzo)
	1. istante in cui è stata proposta
	2. utente offerente (bidder)
	3. meccanismo su aumento sistematico del prezzo
4. categorie
	1. livello della gerarchia ad albero
5. venditori professionali
	1. URL della vetrina online
	2. ulteriori informazioni legali
	3. popolarità (operazione)
			utenti professionali non posson fare bid
6. feeback
	1. voto numerico (0-5)
	2. commento testuale (non obbligatorio)

---
## Diagramma UML delle classi
![[Pasted image 20250428232706.png]]

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
`bid_vincente():Bid`
- precondizioni → sia `b:Bid` tale che esista almeno un link `(this, b):bid_asta`
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `R` l’insieme dei link `(this, b):bid_asta`
		- `result=b` tale che `b.istante` sia il massimo tra tutti i link in `R`

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
- se `p:AstaConclusa` allora `b=p.bid_vincente()` e esista un unico link `(u,b):bid_utente`

### Bid
Un’istanza di questa classe rappresenta una bid

`[Bid.tempo]`
Sia `a:Asta` l’unica associazione coinvolta in `(this,a):bid_asta`
- `a.scadenza>this.istante`
