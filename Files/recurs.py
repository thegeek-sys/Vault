#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:02:26 2021

@author: lizardking
"""

from rtrace import TraceRecursion # thanks Prof. Sterbini!

# %% Fattoriale

# 1. riduzione F(N) -> F(N-1)
# 2. caso base F(1) = 1
# 3. convergenza  N, N-1, ..... 1
# 4. conquer F(N) = N*F(N-1)

@TraceRecursion
def fact(n):
    # 2. caso base
    if n == 1:
        return 1
    # 1. riduzione
    fact_n1 = fact(n-1)
    # 4. conquer
    sol = n * fact_n1
    return sol

# %%% Ex Fattoriale
nfact = fact(5)

# %% Somma primi 1..N numeri ricorsivamente al RITORNO

# 1. riduzione sum(N) -> sum(N-1)
# 2. caso base sum 1 e' 1
# 3. convergenza N....N-1...2...1 (andata, non facciamo niente)
# 4. sum(N) = N + sum(N-1) o anche N + N-1 + sum(N-2) ritorno, componiamo soluzioni

@TraceRecursion
def sumr(n):
    # 2. caso base
    if n == 1:
        return 1
    # riduzione e conquer insieme 3. + 4.
    return n + sumr(n-1)

# %%% Ex sumr
nsum = sumr.trace(5)

# %% Somma primi 1..N numeri ricorsivamente al RITORNO
# con meno chiamate ricorsive

@TraceRecursion
def sumr(n):
    if n == 1:
        return 1
    # 2. caso base
    if n == 2:
        return 3#n + n-1
    # riduzione e conquer insieme 3. + 4.
    return n + sumr(n-1)

# %%% Ex sumr
nsum = sumr.trace(5)


# %% Somma primi 1..N numeri ricorsivamente al RITORNO
# con meno chiamate ricorsive, sempre meno

@TraceRecursion
def sumr(n):
    assert n >=1, 'errore'
    # FIX a lezione:
    # mancava caso 2 e 1
    # se vogliamo arrivare fino ad 1
    # 2. caso base
    if n == 1:
        return 1
    if n == 2:
        return 2 + 1
    if n == 3:
        return 3 + 2 + 1
    # riduzione e conquer insieme 3. + 4.
    return n + sumr(n-1)

# %%% Ex sumr
nsum = sumr.trace(5)

# %% Cosa fa questo codice
@TraceRecursion
def ones(n):
    if n == 1:
        return 1
    return ones(n-1) + 1

# %%% ex ones
one = ones(10)

# %%  PER CASA, Cosa fa questo codice
# cambia il risultato se cambiamo n?
# disegna albero di ricorsione su carta
# per un numero n piccolo, tipo n=3
@TraceRecursion
def whatdoido(n):
    assert n < 10, f'fermati troppo grande {n}'
    if n == 1:
        return 1
    prima = whatdoido(n-1) + 1
    dopo  = -1 + whatdoido(n-1)
    return prima - dopo - 2
# %%% ex ones
rez = whatdoido.trace(5)

# %% PER CASA Somma ricorsivamente i numeri da N a M compresi

def sumr(n, m):
    if m == n:
        return n
    else:
        return m + sumr(n, m-1)

# %% Somma primi 1..N numeri ricorsivamente all'ANDATA

# Ragioniamo in maniera inversa
# invece che ridurre, incrementiamo
# fino ad arrivar alla soluzione

# sommo da 1...N

# 1. incremento i -> i+1
# 2. finisco quando i==n+1 (convergenza e risultato)
# 3. in partenza la somma e' 0, ad ogni passo incremento

def sumrp(i, n, partial_sum=0):
    # 2. convergenza e risultato
    if i == n+1:
        return partial_sum # torniamo il caso generato
    # incremento della soluzione per ogni passo
    # sono ad iterazione i+1 e accumula la somma parziale
    return sumrp(i+1, n, partial_sum=partial_sum+i)
    
# %%% Ex sumr
nsum = sumrp(1, n=5)

# %% Somma primi 1..N numeri ricorsivamente all'ANDATA
# mi risparmio un passo ricorsivo
@TraceRecursion
def sumrp(i, n, partial_sum=0):
    # 2. convergenza e risultato
    if i == n:
        return partial_sum + n # mi risparmio un passo ricorsivo
    return sumrp(i+1, n, partial_sum=partial_sum+i)
    
# %%% Ex sumr

nsum = sumrp.trace(1, n=5)



# %% Stampiamo una stringa ricorsivamente

# 1. s[0] | s[1:]--------
# 2. caso base, un solo carattere rimanente
# 3. convergenza: arrivo a stringa vuota
# 4. conquer: non far niente

def print_rec(S):
    if len(S) == 1:
        print(S[0], end='')
        return 
    print(S[0], end='')
    print_rec(S[1:])
    
# %%% Ex print stringa
print_rec('supercalifragilistichespiralidoso')


# %% Riscrittura equivalente
def print_rec(S):
    # stamp primo char
    print(S[0], end='')
    # se era ultimo ho finito ritorno
    # non faccio altro
    if len(S) > 1:
        # se non era ultimo vado in ricorsione
        print_rec(S[1:])
    # return None
# %%% Ex print stringa
print_rec('supercalifragilistichespiralidoso')


# %% Stampa al RITORNO
def print_rec(S):
    if len(S) > 1:
        # se non era ultimo vado in ricorsione
        print_rec(S[1:])
    # stampo char
    print(S[0], end='')
# %%% Ex print stringa
print_rec('supercalifragilistichespiralidoso')

# %% list

# Cosa fa questo codice?
@TraceRecursion
def process_list(L):
    if not L:
        return []
    
    return process_list(L[:-1]) + [L[-1]]


# %% ex list
L  = process_list.trace([1,2,3])


#%%  PER CASA
# scrivi codice ricorsivo che inverte una lista ricorsivamente
# deve tornare i valori di L ma invertiti
# L = 1, 2 ,3 
# torna 3, 2, 1
def invert(L):
    if not L or len(L) == 1:
        return L
    return invert(L[1:]) + [ L[0] ]
#%%% eval invert

print(invert([1,2,3]))
print(invert([1]))
print(invert([]))
print(invert(['a','z','t','b']))

#%% Invertiamo con indici inversi
@TraceRecursion
def invert(L):
    if not L or len(L) == 1:
        return L # lista
    return [L[-1]] + invert(L[:-1])
           # lista + lista OK

#%%% eval invert

print(invert([1,2,3]))
print(invert([1]))
print(invert([]))
print(invert(['a','z','t','b']))

#%%  inversione con append
# torna 3, 2, 1
def invert(L):
    if not L or len(L) == 1:
        return L # lista
    nL = invert(L[1:]) # lista
    nL.append(L[0]) # appendo valore
    return nL


#%%% eval invert

print(invert([1,2,3]))
print(invert([1]))
print(invert([]))
print(invert(['a','z','t','b']))

#%% Invertiamo con indici inversi con insert
@TraceRecursion
def invert(L):
    if not L or len(L) == 1:
        return L # lista
    nL = invert(L[:-1])
    nL.insert(0,L[-1]) # non ottimale, ogni volta devo aggiungere in cima
                       # [x, n1,----,nN]
    return nL
#%%% eval invert

print(invert([1,2,3]))
print(invert([1]))
print(invert([]))
print(invert(['a','z','t','b']))



#%% Invertiamo ma all' ANDATA
@TraceRecursion
def invert(L, nL=None, idx=0):
    # se non passo valore default e' None
    # quindi inizializzo la lista (sono alla radice)
    if nL is None:
        nL = []
    
    # se le lunghezze sono uguali allora ho finito
    #if len(L) == len(nL):
    if len(L) == idx: # oppure anche quando lungh. di L == idx
        return nL # torno cosa ho generato
    
    # [ i ------ nL]
    nL = [L[idx]] + nL
    return invert(L, nL=nL, idx=idx+1)

#%%% eval invert

print(invert([1,2,3]))
print(invert([1]))
print(invert([]))
print(invert(['a','z','t','b']))


#%% Invertiamo ma all' ANDATA
@TraceRecursion
def invert(L, nL=None):
    # se non passo valore default e' None
    # quindi inizializzo la lista (sono alla radice)
    if nL is None:
        nL = []
        
    # indice corrisponde con lunghezza della 
    # nuva lista
    # idx=0 se nL = []     L[0] non ne ho messo nessuno
    # idx=1 se nL = [x]    L[1] ne ho inserito uno quindi prendo indice 1
    # idx=2 se nL = [x,x], L[2] ne ho inseriti 2 quindi prendo indie 2
    #  perche' si inizia a contare da zero
    idx = len(nL)
    
    # se le lunghezze sono uguali allora ho finito
    #if len(L) == len(nL):
    if len(L) == idx: # oppure anche quando lungh. di L == idx
        return nL # torno cosa ho generato
    
    # [ i ------ nL]
    nL = [L[idx]] + nL
    # adesso il nL prende il nuovo nL come nuovo valore
    return invert(L, nL=nL)

#%%% eval invert

print(invert([1,2,3]))
print(invert([1]))
print(invert([]))
print(invert(['a','z','t','b']))


#%% Immagini Sacchiera ricorsivamente

# voglio generare una immagine a scacchiera
# in maniera ricorsiva.
# Assumo immagine divisibile per 2 ad ogni iterazione ricorsiva
# i.e. dimensione e' una potenza del 2
# assumo immagini quadrate


# voglio rispondere alla domanda, generami una scacchiera
# di dimensioni 128x128

def checkboard(k=1):
    # 2 x 2
    black = (0,)*3
    white = (255,)*3
    # white | black
    row = [white,]*k + [black,]*k
    # white | black
    # black | white
    return [row,]*k + [row[::-1],]*k

    # se k>=3 (ingrandisce il template, rimuove aliasing)
    #     3 times       |     3 times
    # white white white | black black black  
    # white white white | black black black  3 times
    # white white white | black black black   +
    # black black black | white white white  inverto [::-1] per 3 times
    # black black black | white white white    
    # black black black | white white white   
    


# 1. riduzione divido ogni volta per due (perche' poi assemblo per 2)
# 2. caso base immagine di base 2x2 =  k*2 x k*2 con k=1 (k mi serve dopo)
# 3. convergenza  256, 128, 64.....2
# 4. conquer (vedi sotto)

# mi arriva e lo devo ripetere
#       x o
#       o x

# devo ripetere in vericale 2 volte

#       x o
#       o x
#       ---
#       x o
#       o x

# a questo punto  devo ripetere in oriz 2 volte

#       x o | x o
#       o x | o x
#       ---------
#       x o | x o
#       o x | o x

# Nota l'algoritmo e' lineare nel numero di chiamate ricorsive
# perche' ogni volta
# itero contemporaneamente su larghezza ed altezza
# Ho una sola chiamata ricorsiva che ogni volta diminuisce 
# la grandezza di 2 sia in larghezza che altezza

def create_checkboard(n,k):
    assert n%2==0, n
    assert n>=2, 'no checkboard with a single pixel'
    
    if n == 2:
        return checkboard(k)
    
    #
    rows = create_checkboard(n//2, k)
    # concateno le righe
    # x
    # |
    # |
    # x
    rows = rows*2
    # ripeto per orizzontale
    # x ---- x
    # |
    # |
    # x
    return [ r*2 for  r in rows]


# %% run
import images
k = 8 # divido sempre per due ma quando arrivo al caso base genero k*2 |
      # cosi mi vengono piu pixel per il template
      # ogni volta ripet il tempalte 2 volte in oriz e 2 volte in vert.
     
# genero potenze del 2
checks = {2**i : create_checkboard(2**i, k=k) for i in range(1,6)}
# %% viz
for ck in checks.values():
    images.visd(ck)
    
    
    

# %% Palindrome
# Controllare se una stringa è palindroma

# %%% con slice
all_pal = ['madam','osso','ireneneri','salas', 
           'racecar', 'itopinonavevanonipoti',
           'aa','', 'x']

all_non_pal = ['pippo', 'ab', 'supercalifragi', 'sal_l_as']

s = 'madam'
s = 'osso'
s = 'pippo'

print(s[::-1]==s)

# %%% con slice ma meno test
N = len(s)
is_pali = s[N//2+N%2:][::-1] == s[:N//2]

# %%% iterativa con for

def is_palindrom(s):
    N = len(s)
    for i in range(N//2):
        # |0---------N-1|
        if s[i] != s[N-1-i]:
            return False
    return True

# %%% eval pal
print(all(map(is_palindrom, all_pal)))
print(any(map(is_palindrom, all_non_pal)))

# %%%% iterativa con while

def is_palindrom(s):
    start, end = 0, len(s)
    while start < end:
        if s[start] != s[end-1]:
            return False
        start +=1
        end   -=1
    return True

# %%%% Eval pal

print(all(map(is_palindrom, all_pal)))
print(any(map(is_palindrom, all_non_pal)))

# %%%% ricorsiva

def is_palindrom(s, start=0, end=None):
    if end is None:
        end = len(s)
        
    if start >= end:
        return True
    
    if s[start] != s[end-1]:
        return False
    else:
        return is_palindrom(s, start+1, end-1)
    
# %%%% Eval pal   
print(all(map(is_palindrom, all_pal)))
print(any(map(is_palindrom, all_non_pal)))


# %%%% ricorsiva in versione python
# ma copio ogni volta una nuova stringa

@TraceRecursion
def is_palindrom(s):
    if len(s) < 2: # copro singolo char e e vuoto
        return True
    start, end = s[0], s[-1]
    if start != end:
        return False
    else:
        return is_palindrom(s[1:-1])
    
# %%%% Eval pal   
print(all(map(is_palindrom, all_pal)))
print(any(map(is_palindrom, all_non_pal)))


# %% File system come albero binario
from filesystem_as_tree import  FilesystemNode