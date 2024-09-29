---
Created: 2024-09-29
Class: "[[Basi di dati]]"
Related: 
Completed:
---
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

