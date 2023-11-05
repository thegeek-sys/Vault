---
Created: 2023-10-05
Programming language: "[[Python]]"
Related: "[[Slicing]]"
Completed: 
---
---
## Index

1. [[#Introduction|Introduction]]
1. [[#methods|Methods]]
1. [[#Whitespaces|Whitespaces]]
1. [[#Escaping|Escaping]]
1. [[#Immutabilità|Immutabilità]]
1. [[#Slicing|Slicing]]

---
## Introduction
Il tipo `str` (string) in Python che ci permette di rappresentare sequenze di caratteri

```python
type('Ciao') # -> <class 'str'>
type("Ciao") # -> <class 'str'>
print(str(7)) # -> '7'
```
---
## Methods
![[str methods#Methods]]

---
## Whitespaces
Gli **whitespaces** sono delle particolari sequenze di caratteri che ci permettono di formattare l'output in modo particolare (es. andare a capo), per vedere tutti gli whitespace mi basta fare

```python
import string print(string.whitespace) # -> ' \t\n\r\x0b\x0c'

## ES ##
'\t' # tab
'\n' # new_line
print('ab', 2, sep='\t', end=' *') # -> 'ab      2 *'
```
---
## Escaping
Si utilizza il `\` per fare l'**_escape_** (considera il carattere seguente come parte della stringa stessa) in modo tale da evitare problemi derivati dagli apici singoli (`''`) o doppi (`""`) o dai whitespaces, oppure definisco la stringa **raw** (non usabile per apici singoli)

```python
#es
print('Viene usata l\'arancia')
print(r'C:\Windows\system32\cmd.exe') # RAW
print('C:\\Windows\\system32\\cmd.exe')
```
---
## Immutabilità
Le stringhe, a differenza degli altri tipi, sono immutabili e dunque una volta assegnate non possono essere più modificate. L'unico modo per modificarla sarà creare una nuova variabile con lo stesso nome e valore modificato (la sto riassegnando, sposto il link in memoria)

```python
nome = 'Python' nome[0] = 'p' # -> TypeError
nome = nome.lower()
nome = str.lower(nome)
print(nome) # -> 'python'
print('p'+nome[1:]) # -> 'python'
```
---
## Slicing
![[Slicing]]