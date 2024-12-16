---
Created: 2024-12-16
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Approfondimenti - Passwords]]"
Completed:
---
---
## Introduction
Linux usa due file per gestire utenti e le relative password
- `/etc/passwd`
- `/etc/shadow`
Entrambi sono normali file di testo con una sintassi similare, ma hanno funzioni e permessi diversi.

Originariamente esisteva soltanto il file `passwd`, che includeva le password dell’utente in plaintext; nell’implementazione attuale, per ogni riga (utente) in `passwd`, esiste una corrispondente riga in `shadow` che indica la sua password

---
## `/etc/passwd`