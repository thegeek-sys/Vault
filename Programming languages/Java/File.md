---
Created: 2024-05-13
Programming language: "[[Java]]"
Related: 
Completed:
---
---

## Index
- [[#Introduction|Introduction]]
- [[#File|File]]
	- [[#File#File di testo|File di testo]]
	- [[#File#File binario|File binario]]
- [[#FileReader e BufferedReader|FileReader e BufferedReader]]
	- [[#FileReader e BufferedReader#Try with resources|Try with resources]]
	- [[#FileReader e BufferedReader#Con classe Scanner|Con classe Scanner]]
- [[#FileWriter e BufferedWriter|FileWriter e BufferedWriter]]
	- [[#FileWriter e BufferedWriter#Con classe PrintWriter|Con classe PrintWriter]]
- [[#Testi formattati|Testi formattati]]
	- [[#Testi formattati#Scrivere|Scrivere]]
	- [[#Testi formattati#Leggere|Leggere]]
- [[#java.nio.file.Path|java.nio.file.Path]]
	- [[#java.nio.file.Path#Ottenere un Path|Ottenere un Path]]
- [[#Serializzare un oggetto|Serializzare un oggetto]]
	- [[#Serializzare un oggetto#Serial Version UID|Serial Version UID]]
	- [[#Serializzare un oggetto#Leggere un oggetto serializzato|Leggere un oggetto serializzato]]

---
## Introduction
Uno **stream** è un’astrazione derivata da dispositivi dispositivi di input o output sequenziale e i **file** possono essere interpretati come tali (in realtà vengono **bufferizzati** per questioni di efficienza)
Si differenziano in stream di:
- **input** → riceve uno stream di caratteri “uno alla volta”
- **output** → produce uno stream di caratteri

E’ importante ricordare inoltre che gli stream non si applicano solo ai file, ma anche a dispositivi di input/output, internet  ecc.

Per manipolarli in Java esistono 4 differenti classi
- per leggere/scrivere **caratteri** → `java.io.Reader/Writer`
- per leggere/scrivere **byte** → `java.io.StreamInput/StreamOutput`

>[!hint]
>Da Java 5 l’accesso ai file è stato semplificato mediante l’aggiunta della classe `java.util.Scanner`, che però risulta essere **più lenta** perché più potente

>[!warning] Non utilizzare `java.io.File`
>- Molti metodi di File non emettono eccezione quando falliscono
>- Nessun supporto per i collegamenti simbolici
>- Mancanza di metadati: permessi, proprietario, ecc.
>- Molti metodi non scalavano su cartelle piene di file

---
## File
Un **file** è una **collezione di dati salvata** su un supporto di memorizzazione di massa che può essere letto o modificato da programmi differenti. Il programma che lo esegue o modifica deve conoscere il formato dei dati nel file
I file si  distinguono in: file di **testo** e file **binari**

### File di testo
Un file di testo contiene linee di testo (es. in ASCII) ed ognuna di esse termina con un carattere di **nuova linea** (`\n`) o di **carriage return** (`\r`) concatenato con una nuova linea

### File binario
Un file binario può contenere qualsiasi informazione sotto forma di **concatenazione di byte** e diversi programmi potrebbero interpretare lo stesso file in modo diverso (es. immagine)

---
## FileReader e BufferedReader
`BufferedReader` permette una lettura bufferizzata dei caratteri forniti da `FileReader`

```java
BufferedReader br = null;
try {
	br = new BufferedReader(new FileReader(filename));
	
	while(br.ready()) {
		String line = br.readline();
		// ...
	}
}
catch (IOException e) {
	// gestisci l'eccezioe
}
finally {
	if (br != null) {
		try { br.close(); catch(IOException e) {/* gestisci */} }
	}
}
```

### Try with resources
E’ possibile specificare tra parentesi dopo `try` un elenco di istruzioni (separate da `;`) che definiscono **risorse da chiudere automaticamente**

```java
try(BufferedReader br = new BufferedReader(new FileReader(filename));) {
	
	while(br.ready()) {
		String line = br.readline();
		// ...
	}
}
catch (IOException e) {
	// gestisci l'eccezioe
}
```

Questo è possibile solo poiché le classi di questi oggetti implementano `java.lang.AutoCloseable` che è estesa dall’interfaccia `java.io.Closeable`

### Con classe Scanner
Possiamo eventualmente anche leggere file di testo con la classe `java.util.Scanner`
```java
File f = new FIle("mio_file.txt");

try {
	Scanner in = new Scanner(f);
	
	// fineché esiste una prossima riga
	while(in.hasNext())
		// stampa la riga
		System.out.println(in.nextLine())
	in.close();
}
catch(FileNotFoundException e) {
	e.printStackTrace();
}
```

---
## FileWriter e BufferedWriter
Usando il try with resources
```java
try(BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
	bw.write("bla bla bla\n")
	// ...
}
catch(IOException e) {
	// gestisci l'eccezione
}
```

### Con classe PrintWriter
Si può altrimenti usare la classe `PrintWriter` e i metodi corrispondenti
```java
// costruisce l'oggetto File
File f = new File("mio_file.txt")

try {
	// costruisce il Writer di file
	PrintWriter out = new PrintWriter(f);
	
	// scrive due righe di testo
	out.println("Prima riga del file");
	out.println("Seconda riga: ")
	out.print(2);
	
	// chiude il file
	out.close();
}
catch(FileNotFoundException e) {
	e.printStackTrace();
}
```

---
## Testi formattati
### Scrivere
E’ possibile scrivere un file di testo formattato utilizzando la classe `java.util.Formatter`
```java
public class FormattaFile {
	private String nome;
	private int valore;
	
	public void scrivi(String filename) throws IOException {
		Formatter output = new Formatter(filename);
		output.format("%s\t%d, nome, valore");
		output.close();
	}
}
```

### Leggere
Essendo il testo nel file formattato usando dei separatori, lo `Scanner` è in grado di restituire la prossima stringa e il prossimo intero
```java
public class FormattaFile {
	private String nome;
	private int valore;
	
	public void scrivi(String filename) throws IOException {
		Formatter output = new Formatter(filename);
		output.format("%s\t%d, nome, valore");
		output.close();
	}
	
	public void leggi(String filename) throws IOException {
		Scanner input = new Scanner(new File(filename));
		nome = input.next();
		valore = input.nextInt();
	}
}
```

---
## java.nio.file.Path
La classe `java.io.File` è rimpiazzata dall'interfaccia `java.nio.file.Path` che rappresenta un percorso gerarchico

### Ottenere un Path
Per ottenere un Path è possibile utilizzare il metodo `Paths.get`
```java
Path p = Path.get("tmp", "foo");
Path p = Path.get("tmp" + File.separator + "foo");

// è un'abbreviazione per
Path p = FileSystems.getDefault().getPath("tmp/foo")
```


Le operazioni che prima si svolgevano nella classe File ora sono metodi statici della classe `java.nio.file.Files` inclusi metodi di comodo per la creazione di `BufferedReader` e `BufferedWriter`
```java
try(BufferedReader br=Files.newBufferedReader(Path.get("fr.txt"));
	BufferedWriter bw=Files.newBufferedWriter(Path.get("fw.txt"))) {
	// legge da fr.txt e scrive in fw.txt
}
```

---
## Serializzare un oggetto
Serializzare un oggetto vuol dire scrivere in memoria **un oggetto per intero** in modo tale che leggendolo posso riavere l’oggetto per intero senza ulteriori interpretazioni.

>[!warning]
>Solo le classi che implementano l’interfaccia senza metodi `Serializable` sono serializzabili

```java
import java.io.*;

public class OggettoSerializzabile {
	private String nome;
	private int valore;
	
	public OggettoSerializzabile(String nome, int valore) {
		this.nome = nome;
		this.valore = valore;
	}
	
	public void salva(String filename) {
		try {
			// costruisce uno stream output di file
			FileOutputStream fos = new FileOutputStream(filename);
			// costruisce uno stream output di oggetti
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			
			// scrive gli oggetti in formato binario
			os.writeObject(nome);
			os.writeObject(valore);
			
			// chiude lo stream
			os.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		new OggettoSerializzabile("dieci", 10).salva("Obj.ser");
	}
}
```

Quando si serializza un oggetto, tutti gli oggetti a cui esso si riferisce (variabili d’istanza vengono serializzati) e se un solo campo non è serializzabile, l'oggetto non è serializzabile.
Tuttavia è possibile rendere "**transiente**" un campo con la parola chiave `transient`:
- Tale campo non sarà serializzato
- I campi statici sono transient di default

### Serial Version UID
E’ bene specificare sempre un campo `static`, `final` e `long` chiamato `serialVersionUID`, usato in **fase di deserializzazione** per verificare se la versione della classe in uso è la stessa usata per serializzare
```java
public class OggettoSerializzabile implements Serializable {
	// indentificatore univoco di versione
	private static final long serialVersionUID = -1327935836496038L;
	
	private String nome;
	private int valore
	private OggettoSerializzabile next;
}
```

### Leggere un oggetto serializzato
```java
public static OggettoSerializzabile leggi(String filename) {
	try {
		// apertura file
		FileInputStream fis = FileInputStream(filename);
		ObjectInputStream ois = ObjectInputStream(fis);
		
		// lettura
		Object o1 = ois.readObject();
		Object o2 = ois.readObject();
		Object o3 = ois.readObject();
		
		// casting
		String nome = (String)o1;
		int valore = (Integer)o2;
		OggettoSerializzabile next = (OggettoSerializzabile)o3;
		
		// creazione dell'oggetto
		OggettoSerializzabile o = new OggettoSerializzabile(nome,
		valore);
		o.setNext(next)
		
		ois.close()
	}
	catch(ClassNotFoundException e) {
		e.printStackTrace();
	}
	catch(IOException e) {
		e.printStackTrace();
	}
	return o
}
```