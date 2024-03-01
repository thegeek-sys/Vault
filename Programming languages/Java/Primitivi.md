---
Created: 2024-02-29
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index

1. [[#Introduzione|Introduzione]]
2. [[#Variabili|Variabili]]
3. [[#Letterali (o costanti)|Letterali (o costanti)]]
4. [[#Precedenza operatori aritmetici|Precedenza operatori aritmetici]]
---

## Introduzione
Tipi di dati di base built-in.

| Tipo             | Operatori |      Esempio      | Intervallo                               | Space  |
| ---------------- | :-------: | :---------------: | ---------------------------------------- | ------ |
| `byte`           | + - * / % |      27 + 1       | -128…127                                 | 1 byte |
| `short`          | + - * / % |      27 + 1       | -32768...32767                           | 2 byte |
| <u>`int`</u>     | + - * / % |      27 + 1       | -2147483648…2147483647                   | 4 byte |
| `long`           | + - * / % |      27 + 1       | -1e9…1e9                                 | 8 byte |
| `float`          | + - * / % |  3.14 * 5.01e23   | 7 cifre decimali significative           | 4 byte |
| <u>`double`</u>  | + - * / % |  3.14 * 5.01e23   | 15 cifre decimali significative          | 8 byte |
| <u>`boolean`</u> | && \|\| ! |  true \|\| false  | true, false                              | 1 byte |
| <u>`char`</u>    |    + -    |        ‘a’        | Tutti i caratteri codificati con unicode | 2 byte |
| <u>`String`</u>  |    + -    | “Hello” + “World” |                                          |        |
- Le stringhe non sono in realtà tipi primitivi (è in realtà un’array di `char`)
- I char sono interpretati come veri e propri numeri unicode, quindi si utilizzano gli operatori di somma e sottrazione.

---
## Variabili
Una variabile è creata tramite una **dichiarazione**, nella quale deve essere specificato il tipo.

>[!warning]
>Le variabili in Java non hanno un metodo di inizializzazione automatico, il che vuol dire che se definisco una variabile locale ma non gli assegno alcun valore se la provo a chiamare questo mi darà un errore di compilazione

```java
int contatore; // ho inizializzato la variabile contatore di tipo int
```

Il valore viene assegnato attraverso un'**assegnazione**:
```java
contatore = 0;
int numero = 1;
```

> [!info] Static typing
> Il tipo di una variabile (se primitivo) è **statico** - non può cambiare.
> 
> La keyword all’inizio di una dichiarazione`final` rende una variabile non più riassegnabile all’interno del programma

Il nome assegnato a una variabile è il suo **identificatore** (come in python, la prima lettera non può essere un numero) - gli identificatori sono case-sensitive.

>[!info]- Notazione
Si utilizza la **notazione Camel case**:
> - quando si dichiara una variabile composta da più parole, la prima inizia con una minuscola e le successive con maiuscole (es. "contatoreTreniEsplosi")
> - le **classi** devono per forza iniziare con una maiuscola (poi si continua con la camel case)

---
## Letterali (o costanti)
Per letterali si intendono le rappresentazioni a livello di codice sorgente del valore di un tipo di dato. (es. 27 è un letterale per gli interi)

- Le costanti int sono semplici numeri.
- Le costanti long vengono specificate con il suffisso L.
- Le costanti double sono numeri con la virgola (che è un *punto*).
- Le costanti float hanno il suffisso f o F.
- Il prefisso *0b* indica una rappresentazione binaria (es. 0b101 è 5)
- Si può usare un trattino basso per separare le cifre (10_000 == 10000)

---
## Precedenza operatori aritmetici
| Operatori | Precedenza |
| ---- | ---- |
| *, /, % | Valutati per primi da sinistra verso destra |
| +, - | Valutati per secondi da sinistra verso destra |
