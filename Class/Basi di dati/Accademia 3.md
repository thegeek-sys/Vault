---
Created: 2025-05-13
Class: "[[Basi di dati]]"
Related:
---
---
## Definizione dei tipi

```sql
create type strutturato as 
	enum ('Ricercatore', 'Professore Associato', 'Professore Ordinario')
create type lavoro_progetto as 
	enum ('Ricerca e Sviluppo', 'Dimostrazione', 'Management', 'Altro')
create type lavoro_non_progettuale as
	enum ('Didattica', 'Ricerca', 'Missione', 'Incontro Dipartimentale', 'Incontro Accademico', 'Altro')
create type causa_assenza as
	enum ('Chiusura Universitaria', 'Maternità', 'Malattia')
```

---
## Definizione dei domini

```sql
create domain pos_integer as integer
	default 0
	check (value>=0)
create domain stringa_m as varchar(100)
create domain numero_ore as integer
	check (value>=0 and value<=8)
create domain denaro as real
	default 0
	check (value>=0)
```

---
## Creazione tabelle
### Persona
```sql
create table Persona (
	id pos_integer not null,
	nome stringa_m not null,
	cognome stringa_m not null,
	posizione strutturato not null,
	stipendio denaro not null
	primary key (id)
)
```

### Progetto
```sql
create table Progetto (
	id pos_integer not null,
	nome stringa_m not null,
	inizio date not null,
	fine date not null,
	budget denaro not null
	primary key (id),
	unique (nome),
	check (inizio < fine)
)
```

### WP
```sql
create table WP (
	progetto pos_integer not null,
	id pos_integer not null,
	nome stringa_m not null,
	inizio date not null,
	fine date not null,
	primary key (id, nome),
	unique (progetto, nome),
	foreign key (progetto) references Progetto(id),
	check (inizio < fine)
)
```

### AttivitàProgetto
```sql
create table AttivitàProgetto (
	id pos_integer not null,
	persona pos_integer not null,
	progetto pos_integer not null,
	wp pos_integer not null,
	giorno date not null,
	tipo lavoro_progetto not null,
	ore_durata numero_ore not null,
	primary key (id),
	foreign key (persona) references Persona(id),
	foreign key (progetto, wp) references WP(progetto, id)
)
```

### AttivitàNonProgettuale
```sql
create table AttivitàNonProgettuale (
	id pos_integer not null,
	persona pos_integer not null,
	tipo lavoro_non_progettuale not null,
	giorno date not null,
	ore_durata numero_ore not null,
	primary key (id),
	foreign key (persona) references Persona(id),
)
```

### Assenza
```sql
create table Assenza (
	id pos_integer not null,
	persona pos_integer not null,
	tipo causa_assenza not null,
	giorno date not null,
	primary key (id),
	foreign key (persona) references Persona(id),
)
```