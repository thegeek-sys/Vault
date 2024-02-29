## Introduzione
Tipi di dati di base built-in.

| Tipo | Operatori | Esempio | Intervallo |
| ---- | :--: | :--: | ---- |
| `int` | + - * / % | 27 + 1 | -2147483648-2147483647 |
| `double` | + - * / % | 3.14 * 5.01e23 | 15 cifre decimali significative |
| `boolean` | && \|\| ! | true \|\| false | true, false |
| `char` | + - | ‘a’ | Tutti i caratteri codificati con unicode |
| `String` | + - | “Hello” + “World” |  |
- Le stringhe non sono in realtà tipi primitivi.
- I char sono interpretati come veri e propri numeri unicode, quindi si utilizzano gli operatori di somma e sottrazione.

## Variabili
Una variabile è creata tramite una **dichiarazione**, nella quale deve essere specificato il tipo:
```java
int contatore; // ho inizializzato la variabile contatore di tipo int
```

Il valore viene assegnato attraverso un'**assegnazione**:
```java
contatore = 0;
int numero = 1;
```

> [!warning] Static typing
> Il tipo di una variabile (se primitivo) è **statico** - non può cambiare.
> La keyword all’inizio di una dichiarazione`final` rende una variabile non più riassegnabile all’interno del programma

Il nome assegnato a una variabile è il suo **identificatore** (come in python, la prima lettera non può essere un numero) - gli identificatori sono case-sensitive.

>[!info]- Notazione
Si utilizza la **notazione Camel case**:
> - quando si dichiara una variabile composta da più parole, la prima inizia con una minuscola e le successive con maiuscole (es. "contatoreTreniEsplosi")
> - le **classi** devono per forza iniziare con una maiuscola (poi si continua con la camel case)

## Letterali (o costanti)
Per letterali si intendono le rappresentazioni a livello di codice sorgente del valore di un tipo di dato. (es. 27 è un letterale per gli interi)
##### costanti intere e in virgola mobile
- Le costanti int sono semplici numeri.
- Le costanti long vengono specificate con il suffisso L.
- Le costanti double sono numeri con la virgola (che è un *punto*).
- Le costanti float hanno il suffisso f o F.
- Il prefisso *0b* indica una rappresentazione binaria (es. 0b101 è 5)
- Si può usare un trattino basso per separare le cifre (10_000 == 10000)
#### precedenza operatori aritmetici
![[Screen Shot 2024-02-28 at 09.39.14.png]]
come in matematica.

#### caratteri e stringhe
I char seguono la **codifica unicode** (basata su interi a 16 bit), e sono racchiusi da apici (singoli) - 'a'.
>[!caratteri di escape]-
>- '\t' - tab
>- '\n' - a capo
>- '\\' - backslash
>- ' \ ' ' - apice
>- '\"' - virgolette
#### operatori
**incrementi**:
- var++ (var = var +1)
- var--
 
diversi da:
- ++var, --var
 
*pre vs post-incremento*:
a++ ha come risultato a, e poi lo incrementa di 1.

```java
int a = 3;
int c = a++
```
qui, c vale 3 (il compilatore dà prima a c il valore di a, e poi aumenta a di 1)

```java
int a = 3;
int c = ++a
```
qui, c vale 4 (e anche a).

quindi:
```java
int a = 4;
int c = 3;
int z = (a++) - (c--);
```
 prima z = 1 
poi a diventa 5 e c diventa 2

**operatori booleani**:
- && - and logico 
- || - or
- ! - not
- ^ - xor

&  e | - and  e or bit a bit (per i binari)
 
**relazionali**:
- ==
- !=
- < , <= , > , >=
- istanceof

**operatore ternario**:
- ? :
 
**shift**:
- <<,  >>, >>>
utili per i numeri binari: ogni shift a sinistra moltiplica per 2 (aggiungo uno 0 a destra in un numero binario)


