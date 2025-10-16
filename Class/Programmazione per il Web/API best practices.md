---
Class: "[[Programmazione per il Web]]"
Related:
  - "[[REST]]"
---
---
## URIS

>[!warning]
>Le URI rappresentano risorse, non azioni

Per questo motivo nelle URI bisogna usare i sostantivi (identificano oggetti) e mai includere verbi (l’azione è definita dal metodo HTTP)

| Metodo   | Esempio (corretto)      | Esempio (errato)              |
| -------- | ----------------------- | ----------------------------- |
| `GET`    | `/managed-devices/{id}` | `/get-managed-device/{id}`    |
| `PUT`    | `/managed-devices/{id}` | `/update-managed-device/{id}` |
| `POST`   | `/users`                | `/create-user`                |
| `DELETE` | `/managed-devices/{id}` | `/remove-managed-device/{id}` |
Per convenzione si usano sostantivi *singolari* per le singole risorse e *plurali* per le collezioni

>[!example]
>- singolo: `/users/admin`
>- collezione: `/users`

---
## Struttura e gerarchia
