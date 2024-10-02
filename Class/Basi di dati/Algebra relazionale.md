---
Created: 2024-09-29
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
L’algebra relazione è un linguaggio **formale** per interrogare un database relazionale: consiste di un insieme di operatori che possono essere applicati a una (operatori unari) o due (operatori binari) istanze di relazione e forniscono come risultato una nuova istanza di relazione (che può essere “salvata” in una ”variabile”).
Ma è anche un linguaggio **procedurale** in quando l’interrogazione consiste in un’espressione in cui compaiono operatori dell’algebra e istanze di relazioni della base di dati, in una sequenza che prescrive in maniera puntuale l’ordine delle operazioni e i loro operandi

---
## Proiezione
La **proiezione** consente di effettuare un taglio verticale su una relazione cioè di selezionare solo alcune colonne (attributi).
$$
\pi_{\text{A1, A2, } \dots, \text{ Ak}}(r)
$$
Seleziona quindi le colonne di $r$ che corrispondono agli attributi $\text{A1, A2, }\dots, \text{Ak}$

### Esempio
![[Screenshot 2024-09-26 alle 15.40.18.png|center|500]]

> [!warning] Attenzione
> Si seguono le regole insiemistiche. Nella relazione risultato **non** ci sono **duplicati**.
> Se vogliamo conservare i clienti omonimi dobbiamo aggiungere un ulteriore attributo in questo caso la **chiave**
> ![[Screenshot 2024-09-26 alle 15.42.26.png|400]]

---
## Selezione
La **selezione** consente di effettuare un “taglio orizzontale” su una relazione, cioè di selezionare solo le righe (tuple) che soddisfano una data condizione
$$
\sigma_{\text{C}}(r)
$$
Seleziona le tuple di $r$ che soddisfano la condizione $\text{C}$ la quale è un’espressione booleana composta in cui i termini semplici sono del tipo:
- $\text{A }\theta \text{ B}$
- $\text{A }\theta \text{ 'nome'}$
dove:
- $\theta$ → un operatore di confronto ($\theta \in \{<, =, >, \leq, \geq\}$)
- A e B → due attributi con lo stesso dominio ($\text{dom(A) = dom(B)}$)
- nome → un elemento di $\text{dom(A)}$ (costante o espressione)

### Esempio
![[Screenshot 2024-09-29 alle 17.35.02.png|center|550]]


---
## Unione
L’unione serve a costruire una relazione contenente tutte le ennuple che appartengono ad almeno uno dei due operandi
$$
r_{1} \cup r_{2}
$$

> [!warning]
> L’unione può essere applicata a due istanze **union compatibili**, ovvero solo se:
> 1. hanno lo stesso numero di attributi
> 2. gli attributi ordinatamente (corrispondenti) sono definiti sullo stesso dominio
> 3. ordinatamente hanno lo stesso significato (es. matricola ≠ numero di telefono)

### Esempio
![[Screenshot 2024-10-02 alle 15.37.22.png|440]]
sono union compatibili
$$
\text{Personale}=\text{Docenti}\cup \text{Amministrativi}
$$
![[Screenshot 2024-10-02 alle 15.38.51.png|440]]


![[Screenshot 2024-10-02 alle 15.43.21.png|440]]
In questo caso non posso fare l’unione (in $\text{Amministrativi}$ ci sta un attributo in più). Per risolvere dunque devo prima fare una proiezione per poter poi fare l’unione. (non era necessario fare la proiezione sui docenti)
$$
\text{Personale}=\text{Docenti}\cup \pi_{\text{Nome, CodDoc, Dipartimento}}(\text{Amministrativi})
$$
![[Screenshot 2024-10-02 alle 15.49.56.png|440]]
