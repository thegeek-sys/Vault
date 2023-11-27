---
Created: 2023-11-21
Programming language: "[[Python]]"
Related:
  - "[[Matrix]]"
  - "[[Shallow and Deep Copy]]"
---
---
## Introduction
Le immagini possono essere:
- **raster** → griglia che quantizza i colori (formati: `jpeg`, `png`, `tiff`)
	- Possono essere rappresentate come:
		- Scala di grigi → lista di liste di valore in scala di grigi
		- RGB → lista di liste di tuple che indicano i colori RGB
- **vettoriali** → curve e tracciati matematici che descrivono l’immagine (formati: `svg`, `eps`, `pdf`)


![[matrix rgb.png|150]]
In realtà le immagini RGB più formalmente possono essere viste come:
- un tensore di dimensioni `HxWx3`
- una matrice di profondità 3

![[sistema riferimento rgb.png|300]]
Il loro sistema di riferimento è ordinato da sinistra a destra e dall’alto al basso (lo 0,0 si trova in alto a sinistra)
Installazione librerie
```python
get_ipython().system(' wget https://twiki.di.uniroma1.it/pub/Programmazione1/AA20_21/DiarioDelleLezioni-CanaleAL/png.py.txt &>/dev/null')
get_ipython().system(' mv png.py.txt png.py')
import png
import io 

# from Prof. Andrea Sterbini
class Image:                                                                                                                    
    '''Oggetto che contiene una immagine come lista di liste di colori (R,G,B) e che viene                                         
    direttamente visualizzate in IPython console/qtconsole/notebook col metodo _repr_png_'''  
    
    def __init__(self, img, mode='RGB'):                                                                                                       
        self.pixels = img  
        self.mode = mode

    def _repr_png_(self):                                                                                                          
        '''Produce la rappresentazione binaria della immagine in formato PNG'''                                                    
        if self.pixels:
            img = png.from_array(self.pixels, self.mode)                                                                                   
            b = io.BytesIO()                                                                                                           
            img.save(b)                                                                                                                
            return b.getvalue()

get_ipython().system('wget https://twiki.di.uniroma1.it/pub/Programmazione1/AA20_21/DiarioDelleLezioni-CanaleAL/images.py.txt &>/dev/null')
get_ipython().system(' mv images.py.txt images.py')
import images
```

---
##  Disegnare su immagini

### Linee

```python
colormap = {'red': (255,0,0), 'blu':(0,0,255), 'green':(0,255,0),
			'black':(0,0,0), 'white':(255,255,255)}
# mi è comodo per definire dei colori standard
color_mat = create_matrix_lc(256,256, colormap['black'])

'''
N.B. ricordarsi  che  mat è indicizata [riga][colonna] oppure [y][x]
'''

# per scrivere su una riga per x+lenght
def plot_line_h(mat, x, y, lenght, value):
	# selezioniamo la riga ci scriviamo sopra da x a x+lenght
	mat[y][x:x+lenght] = [value] * lenght

# per scrivere su più righe lungo tutto x
def plot_line_w(mat, x, y, lenght, value):
	# ci muoviamo da y fino a y+lenght, x rimane fissa
	for each_y in range(y,y+lenght):
		mat[each_y][x] = value
```

### Rettangoli

```python
def  plot_rect(mat, x, y, Wr, Hr, value, clip=False):
	'''
	plottiamo il rettangolo:
	1. upper segment
	2. lower segment
	3. left segment
	4. right segment
	
	# x,y -------------- x+Wr-1,y
    # |                     |
    # |                     |                                          
    # |                     |                     
    # |                     |                     
    # x,y+Hr-1,---------x+Wr-1,y+Hr-1
	'''
	del clip(v, min_v, max_v):
		return min(max(min_v, v), max_v)
	
	H = len(mat)
	W = len(mat[0])
	
	# per ridimensionare il rettangolo se va in overflow rispetto
	# ai margini dell'immagine
	if clip:
		x, y = clip(x, 0,  W-1), clip(y, 0, H-1)
		w, h = clip(w, 0,  W-1-x), clip(h, 0, H-1-y)

	# plotting
	plot_line_h(mat, x,      y,      Wr, value) # 1.
	plot_line_h(mat, x,      y+Hr-1, Wr, value) # 2.
	plot_line_v(mat, x,      y,      Hr, value) # 3.
	plot_line_v(mat, x+Wr-1, y,      Hr, value) # 4.

# un secondo modo per evitare di sbordare
def draw_pixel2(img, x, y, colore):
	altezza = len(img)
	larghezza = len(img[0])
	if 0 <= x < larghezza and 0 <= y < altezza:
		img[y][x] = colore
```

#### Altri
```python
# rettangoli concentrici
H = 100
W = 50
black = create_matrix(H, W)
step = 2
den = 2
size = min(H,W)
for p in range(0,size//den,step):
    plot_rect(black, p, p, W-p-p, H-p-p, colormap['white'])
images.visd(black)


# rettangolo pieno
black = create_matrix(H, W)
def fill_rect(im, x, y, Wr, Hr, col):
    H, W = shape(im)
    # x,y -------------- x+Wr-1,y
    # |-------------------- |
    # |                     |                                          
    # |                     |                     
    # |                     |                     
    # x,y+Hr-1,---------x+Wr-1,y+Hr-1
    for delta_h in range(Hr):
        draw_h_line(im, x, y+delta_h, Wr, col, W, H)
fill_rect(black, 0, 0, W, H, colormap['yellow'])
images.visd(black)


# rettangoli concentrici pieni
black = create_matrix(H, W)
step = 2
keys = list(colormap.keys())
N = len(keys)
size = min(H,W)
for i, p in enumerate(range(0,size//2,step)):
    fill_rect(black, p, p, W-2*p, H-2*p, colormap[keys[i%N]])
images.visd(black)
```

---
## Aprire immagini

```python
import images
im = images.load('gopttp_small.png')

# scrivere riga nera al centro dell'immagine
R = len(im)
W = len(im[0])
im[H//2] = [(0,)*3]*W
images.save(im, 'giotto_edited.png')

images.visd('giotto_edited.png') # mi permette di fare un render direttamente in sypder su iphyton

```

---
## Flippare immagini
### Asse verticale

```python
def flip_v(img):
    '''
    Flipping the image wrt to the vertical axis
    in a functional way.
    More complex: img refers to <list of <list>> as [ row_0, ...., row_n-1]
    I can pass the iterator img to map that will see each item as a row
    then I define a lambda function that takes the row and flips it
    either with [::-1] or I could have used reversed()
    '''
	H = len(img)
	W = len(img[0])
	flipped_img = []
	for each_y in range(H):
		# sto invertendo ogni colonna
		flipped_img.append(img[each_y][::-1])
	return flipped_img

def flip_v_map(img):
	H = len(img)
	W = len(img[0])
	flipped = map(lambda each_r: each_r[::-1], img)
	return  list(flipped)
```

### Asse orizzontale

```python
def flip_h(img):
	'''
    Img refers to <list of <list>> as [ row_0, ...., row_n-1].
    What we have to do is simply "reshuffle" the order of the 
    rows to follow the reverse order.
    So what I can do is to treat the img as an iterator that I can
    just reverse immediatly. The function that process each item
    will return in the correct orect just the row.
    Note: we optimized the function more to avoid calling the unused map
    
    [ r0
      r1
      ...
      rn-1
    ]
    
    becomes
    
    [ rn-1
      r0
      ...
      r1
    ]
    '''
	H = len(img)
	W = len(img[0])
	flipped_img = []
	for each_y_flip in reversed(range(H)): # range(H-1,-1,-1)
		# sto invertendo ogni riga
		flipped_img = [row[::-1] for row in im]
	return flipped_img

def flip_h_map(img):
	H = len(img)
	W = len(img[0])
	return list(map(lambda each_rr: each_rr, reversed(img)))
```

---
## Shape di una matrice
```python
def shape(mat):
    # immediatly check empty matrix
    if len(mat) == 0:
        return 0, 0
    if len(mat[0]) == 0:
        return 1, 0
    # rows corresponds to height
    r = len(mat)
    c = len(mat[0])
    return r, c
```

---
## Ruotare immagini
### Sinistra
```python
#          Sorgente
#    [c_0-- r_0 ------c_n-1]
#    [------ r_1 ----------]
#             .....
#    [------ r_n-1 --------]

# Destinazione (ragioniamo sulla riga i-esima)

# [c_0-- r_0 ------c_n-1] diventa la colonna

# [c_n-1
# ...
# ....
# c_0]

def rotate_left(im):
    return [ [im[r][c] for r in range(H)] for c in reversed(range(W))]
```

### Destra
```python
def rotate_right(im):
      return [ [im[r][c] for r in range(H)] for c in range(W)]
```

---
## Crop immagini

```python
def crop(im, x, y, w, h):
    H, W = shape(im)
    return[ [c for c in row[x:x+w]] for row in im[y:y+h]]
images.visd(crop(im,0,0,W//2,H//2))
H,W = shape(im)
```

---
## Scala di grigi

```python
'''
Faccio la media di tre colori del singolo pixel e assegno il valore medio a tutti e tre i colori es. (156, 99, 38) → (98, 98, 98)
'''
def gray(im):
    return [[ (sum(c)//3,)*3 for c in row] for row in im ]
```

---
## Applicare filtri

```python
# Applico un filtro generico (passo una funzione)
def filtro_null(pix):
    return pix

def filter_im(im, filter_func):
    return [[ filter_func(c) for c in row] for row in im ]

images.visd(filter_im(im,filtro_null))
```

### Filtro luminosità
```python
# Applico un filtro che aumenta intensità
def luminosita(pix,k):
    def clip(p):
        return max(min(int(round(p*k)),255),0)
    return tuple(clip(p) for p in pix)
```

### Shuffling
```python
# Aggiungo rumore all'immagine
import random
def shuffle(im,x,y, H, W, size=20):
    rx = random.randint(-size, size)
    ry = random.randint(-size, size)
    xx = x + rx
    yy = y + ry
    pix = im[y][x]
    if 0 <= xx < W and 0 <= yy < H:
        pix = im[yy][xx]
    return pix
```

### Blur
```python
from tqdm import tqdm
def blur(im,x, y, H, W, k=5):
    # k=1;x=0 si fa -1, 0, +1 compreso
    somma = 0, 0, 0
    count = 0
    for xx in range(x-k,x+k+1):
        for yy in range(y-k,y+k+1):
            if 0 <= xx < W and 0 <= yy < H:
                pix = im[yy][xx]
                somma = tuple(map(lambda s,p: s+p, somma,pix))
                count += 1 
    return tuple(map(lambda s: min(max(s//count,0),255), somma))
    
    
def filter_im(im, filter_func):
    H, W = shape(im)
    return [[ filter_func(im,x,y, H, W) for x, c in enumerate(row)] 
            for y, row in tqdm(enumerate(im),desc='blurring',total=H) ]
images.visd(filter_im(im,blur))
```

---
## Compressione immagini
- **lossless** → è possibile ricondursi al file di partenza
- **lossy** → .jpg viene compressa un’immagine perdendo il file originale

## Tipico esercizio esame
- immagine da parsare e restituire un rettangolo, segmento etc e restituire parametrizzazione di questo
- immagine da disegnare