---
Created: 2025-03-04
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Index
- [[#La shell|La shell]]
- [[#Il prompt|Il prompt]]
- [[#Il comando|Il comando]]
- [[#La “history” della bash|La “history” della bash]]
- [[#Gli utenti|Gli utenti]]
- [[#`sudo`|sudo]]
- [[#Creazione di altri utenti|Creazione di altri utenti]]
- [[#Cambiare utente|Cambiare utente]]
---
## La shell
La **shell** è un programma che esegue altri comandi, spesso chiamato terminale. Esistono vari tipi di shell come ad esempio:
- Thompson/Bourne shell → sh
- Bourne-Again shell → bash
- KornShell → ksh

---
## Il prompt
La bash scrive un prompt ed attende che l’utente scriva un “comando”. Il prompt tipico è così costituito
```bash
nomeutente@nomemacchina:~cammino$
```
Dove `cammino` è il path della directory home alla directory attuale (quindi se si è semplicemente nella home, c’è solo `~`). Se la directory corrente non si trova nel sottoalbero radicato nella home, allora il `cammino` è il path assoluto

---
## Il comando
Ogni comando verrà eseguito come segue:
```bash
comando [opzioni] argomenti obbligatori
```

Le opzioni tipicamente hanno una doppia segnatura che può essere `--parola` oppure `-carattere`. Ad esempio per il comando `cp` si ha:
- `-i` → `--interactive`
- `-r` → `--recursive`

Le opzioni inoltre possono avere un argomento, che può essere indicato in varie maniere:
- `-k1`
- `-k 1`
- `--key=1`

Inoltre le opzioni senza argomento sono raggruppabili:
- `-b -r -c` → `-brc`

---
## La “history” della bash
Con i tasti freccia su e freccia giù è possibile scorrere la lista dei comandi dati. Una volta utilizzato il comando selezionato, questo può essere modificato.
Inoltre è possibile ricercare un comando data una certa keywork con la combinazione di tasti `CTRL+r`

---
## Gli utenti
Durante l’installazione di linux è necessario specificare (almeno) un utente (alcune versioni creano un utente automaticamente)
Nonostante ciò non tutti gli utenti possono fare login, ad esempio tipicamente `root` non può fare login ma un utente può acquisire i diritti di `root` mediante il comando `su` e `sudo`

Ogni utente appartiene almeno ad un gruppo automaticamente creato con lo stesso nome dell’utente principale. Per poter accedere ai gruppi di un utente è necessario il comando `groups [nomeutente]`

---
## `sudo`
Nelle distribuzioni della famiglia Ubuntu l’utente principale è un `sudoer`, ovvero appartiene al gruppo predefinito `sudo`. Gli utenti appartenenti a questo gruppo possono eseguire comandi da `root` usando il comando `sudo comando`.

---
## Creazione di altri utenti
Per creare nuovi utenti si usa il comando `adduser nuovoutente` (di default l’utente creato non appartiene al gruppo `sudo`). Per aggiungere un utente ad un gruppo si usa il comando `adduser nuovoutente gruppo`

---
## Cambiare utente
Per cambiare utente si usa il comando `su [- | -l | --login] nomeutente` (se si esegue `su -`, si fa in automatico il login all’utente `root`)
