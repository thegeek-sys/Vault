---
Created: 2024-09-29
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Index
- [[#Introduzione|Introduzione]]
- [[#Definizioni|Definizioni]]
- [[#Relazioni e tabelle|Relazioni e tabelle]]
	- [[#Relazioni e tabelle#Esempio|Esempio]]
- [[#Valori nulli|Valori nulli]]
- [[#Vincoli di integrità|Vincoli di integrità]]
	- [[#Vincoli di integrità#Vincoli  intrarelazionali|Vincoli  intrarelazionali]]
		- [[#Vincoli  intrarelazionali#Esempio|Esempio]]
	- [[#Vincoli di integrità#Vincoli interrelazionali|Vincoli interrelazionali]]
		- [[#Vincoli interrelazionali#Esempio|Esempio]]
- [[#Chiavi|Chiavi]]
- [[#Dipendenza funzionale|Dipendenza funzionale]]
	- [[#Dipendenza funzionale#Esempio|Esempio]]
---
## Introduzione
Il modello relazione venne proposto per la prima volta da Codd nel 1970 per favorire l’indipendenza dei dati, ma fu disponibile in DBMS reali solo a partire dal 1981.
Questo modello è basato sulla nozione matematica di **relazione** le quasi si traducono in maniera naturale in **tabelle** (infatti useremo sempre il termine relazione invece di tabella). Dati e relazioni (riferimenti) tra dati di insiemi (tabelle) diversi sono rappresentati come **valori**

---
## Definizioni
Il **dominio** un insieme possibilmente infinito di valori (es. insieme dei numeri interi, insieme delle stringhe di caratteri di lunghezza 20 ecc.). Siano $\text{D1,D2,}\dots \text{Dk}$ domini, non necessariamente distinti. Il prodotto cartesiano di tali domini è denotato da:
$$
\text{D1} \times \text{D2}\times\dots \times \text{Dk}
$$
è l’insieme
$$
\{(\text{v1, v2, }\dots \text{vk})|\text{v1}\in \text{D1, } \text{v2}\in \text{D2,} \dots \text{Vk} \in \text{Dk}\}
$$

Una **relazione matematica** è un qualsiasi sottoinsieme del prodotto cartesiano di uno o più domini
Una relazione che è sottoinsieme del prodotto cartesiano di k domini si dice di **grado k**
Gli elementi di una relazione sono detti **tuple**. Il numero di tuple in una relazione è la sua cardinalità. Ogni tupla di una relazione di grado k ha k componenti ordinate ma non c’è ordinamento tra le tuple

> [!info] Esempio
> - supponiamo $k=2$
> - $\text{D1} = \{\text{bianco}, \text{nero}\}, \text{D2} = \{0,1,2\}$
> - $\text{D1}\times \text{D2} = \{(\text{bianco, }0), (\text{bianco, }1), (\text{bianco, }2), (\text{nero, }0), (\text{nero, }1), (\text{nero, }2)\}$
> 
> $\{(\text{bianco, }0), (\text{nero, }0), (\text{nero, }2)\}$ sono una relazione di grado 2, cardinalità 3 e con tutple $(\text{bianco, }0), (\text{nero, }0), (\text{nero, }2)$
> $\{(\text{nero, }0), (\text{nero, }2)\}$ è una relazione di grado 2, cardinalità 2 e con tuple $(\text{nero, }0), (\text{nero, }2)$

---
## Relazioni e tabelle
Però a questo punto in che modo posso interpretare i dati nella tabella?
Utilizzo una notazione anche per la tabella e le colonne
Infatti uso:
- un **attributo** è definito da un nome $A$ (che ne descrive il ruolo) e dal **dominio** dell’attributo a che indichiamo con $\text{dom(A)}$ (la coppia dominio, nome di attributo è definita attributo)
- sia R un insieme di attributi. Un’ennupla su $R$ è una **funzione** definita su R che associa ad ogni attributo $A$ in $R$ un elemento di $\text{dom(A)}$
- se t è un’ennupla su $R$ ed $A$ è un attributo in $R$, allora con $t(A)$ indicheremo il valore assunto dalla funzione $t$ in corrispondenza dell’attributo $A$
- con **schema di relazione** l’rappresento l’insieme degli attributi di una relazione $R(\text{A1, A2, }\dots \text{, Ak})$ (es. $\text{Info\_Città(Città, Regione, Popolazione)}$ ). Questo rimane invariato nel tempo e descrive la struttura stessa della relazione. Per **schema di base di dati** si intende un insieme di schemi di relazione con nomi differenti
- per **istanza** di una relazione con schema $R(X)$ si indica l’insieme $R$ di tuple su $X$. Questa contiene i valori attuali, che possono cambiare molto rapidamente nel tempo (corpo della relazione)
- con $\text{t[Ai]}$ si indica il **valore dell’attributo** con nome $\text{Ai}$ della tupla $\text{t}$ (nell’esempio sotto se $\text{t}$ è la seconda tupla, allora $\text{t[Cognome] = Bianchi}$)
- se $\text{Y}$ è un sottoinsieme di attributi dello schema $\text{X}$ di una relazione allora $\text{t[Y]}$ è il sottoinsieme di valori nella tupla $\text{t}$ che corrispondono ad attributi contenuti in $\text{Y}$. Questo è chiamato **restrizione** di $\text{t}$

### Esempio
![[Screenshot 2024-09-29 alle 15.58.01.png|430]]
![[Screenshot 2024-09-29 alle 16.04.12.png|430]]

---
## Valori nulli
I valori **NULL** rappresentano la mancanza di informazione o il fatto che l’informazione non è applicabile. Questo valore può essere assegnato a un qualunque dominio, indipendentemente da come è definito.
Tutti i valori NULL sono considerati diversi tra di loro (un valore NULL nel campo di una tupla è diverso dal valore NULL di un altro campo di una stessa tupla oppure dello stesso campo di un’altra tupla)

---
## Vincoli di integrità
I **vincoli di integrità** sono delle proprietà che devono essere soddisfatte da ogni istanza della base di dati (sono **legate allo schema**). Questi descrivono proprietà specifiche del campo di applicazione, e quindi delle informazioni ad esso relative modellate attraverso la base di dati.
Una istanza di base di dati è corretta se soddisfa tutti i vincoli di integrità associati al suo schema

Esistono due tipi di vincoli:
- **Vincoli intrarelazionali** → definiti sui valori di singoli attributi (di dominio) o tra valori di attributi di una stessa tupla o tra tuple della stessa relazione
- **Vincoli interrelazionali** → definiti tra più relazioni

### Vincoli  intrarelazionali
Questi possono essere:
- Vincolo di chiave primaria (**primary key**) → unica e mai nulla
- Vincoli di dominio (es. ASSUNZIONE > 1980)
- Vincoli di unicità (**unique**)
- Vincoli di esistenza del valore per un certo attributo (**not null**)
- Espressioni sul valore di attributi della stessa tupla (es. data_arrivo < data_partenza)
#### Esempio
![[Screenshot 2024-09-29 alle 16.38.48.png|500]]

![[Screenshot 2024-09-29 alle 16.39.44.png|500]]

**Vincoli di dominio**
- ASSUNZIONE > 1980
- (Voto ≥ 18) AND (Voto ≤ 30)

**Vincoli di tupla**
- (Voto = 30) OR NOT (Lode = “si“)

**Vincoli tra valori in tuple di relazioni diverse**
- DIP REFERENCES DIPARTIMENTO.NUMERO
- Studente REFERENCES Studenti.Matricola

### Vincoli interrelazionali
- Vincolo di integrità referenziale (**foreign key**) → quando porzioni di informazione in relazioni diverse sono correlate attraverso valori di chiave 
#### Esempio
l’attributo Vigile della relazione INFRAZIONI e l‘attributo Matricola (chiave) della relazione VIGILI
![[Screenshot 2024-09-29 alle 17.02.40.png|430]]

gli attributi Prov e Numero di INFRAZIONI e gli attributi Prov e Numero (chiave) della relazione AUTO
![[Screenshot 2024-09-29 alle 17.07.31.png|430]]

---
## Chiavi
Una **chiave** di una relazione (non necessariamente unica) è un attributo o insieme di attributi (chiave **composta**) che identifica univocamente una tupla
Un attributo per essere considerato una chiave deve rispettare queste condizioni:
1. per ogni istanza di una relazione $\text{R}$, non esistono due tuple distinte $\text{t1}$ e $\text{t2}$ che hanno gli stessi valori per gli attributi in un insieme $\text{X}$ (chiavi), tali cioè che $\text{t1[X] = t2[X]}$
2. nessun sottoinsieme proprio di $\text{X}$ soddisfa la prima condizione

Una relazione potrebbe avere inoltre più chiavi  alternative. Quella più usata o quella  composta da numero minore di attributi viene scelta come chiave **primaria**. La chiave primaria non ammette valori nulli e ne deve esistere almeno una all’interno di ogni relazione (sono infatti le chiavi a consentire di mettere in relazione dati in tabelle diverse)

Si parla di chiave **minimale** quando una chiave non contiene un sottoinsieme di attributi che a sua volta è una chiave (si applica ai sottoinsiemi di super-chiave).
Si parla di **super-chiave** quando un insieme di attributi contiene una chiave (una chiave è in senso improprio una super-chiave ma non il contrario)

---
## Dipendenza funzionale
Una **dipendenza funzionale** stabilisce un particolare legame semantico tra due insiemi non-vuoti di attributi $\text{X}$ e $\text{Y}$ appartenenti ad uno schema $\text{R}$
Tale vincolo si scrive $\text{X} \rightarrow \text{Y}$ e si legge $\text{X}$ determina $\text{Y}$

diremo che una relazione $r$ con schema $R$ **soddisfa** la dipendenza funzionale $X \rightarrow Y$ se:
1. la dipendenza funzionale $X \rightarrow Y$ è applicabile ad $R$, nel senso che sia $X$ sia $Y$ sono sottoinsiemi di $R$
2. le ennuple in $r$ che concordano su $X$ concordano anche su $Y$, cioè per ogni coppia di ennuple $t1$ e $t2$ in $r$ (se hanno la stessa $X$ devono avere la stessa $Y$)
	$$
	t1[X] = t2[X] \rightarrow t1[Y] = t2[Y]
	$$


### Esempio
Supponiamo di avere uno schema di relazione
$$
\text{VOLI(CodiceVolo, Giorno, Pilota, Ora)}
$$
Con i vincoli:
- un volo con un certo codice parte sempre alla stessa ora
- esiste un solo volo con un dato pilota, in un dato giorno, ad una data ora

I vincoli corrispondono alle dipendenze funzionali:
- $\text{CodiceVolo} \rightarrow \text{Ora}$
- $\text{\{Giorno, Pilota, Ora\}} \rightarrow \text{CodiceVolo}$
- $\text{\{CodiceVolo, Giorno\}} \rightarrow \text{Pilota}$
