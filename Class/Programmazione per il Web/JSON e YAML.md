---
Created: 2025-10-09
Class: "[[JSON e YAML]]"
Related:
---
---
## JSON
Il **JSON** (*JavaScript Object Notation*) è un formato di testo leggero per lo scambio di dati basato su un sottoinsieme di Javascript (es. dati inviati con richieste e risposte RESTful). Viene anche utilizzato per archiviare dati

```json
{
	"user": {
		"firstName": "John",
		"lastName": "Smith",
		"age": 27
	}
}
```

### Oggetti e array
JSON utilizza solo due concetti familiari:
- **object**
- **array**

Questi risultano facili da leggere e scrivere per gli umani e sono facili da analizzare e generare per le macchine

#### Oggetto
Collezione non ordinata di coppie `nome:valore` racchiusa tra parentesi graffe
- `nome` → stringa tra `""`, unica all’interno dell’oggetto
- `valore` → numero, stringa, booleano, null, array, object

```json
{
	"WASA": {
		"name": "Web and Software Architecture",
		"semester": 1
	}
}
```

#### Array
Elenco ordinato di valori, separato da virgole, racchiuso tra parentesi quadre

```json
{
	"wasWeekdays": ["tuesday", "thursday"]
}
```

>[!example]- Esempio JSON
>
>```json
>{
>	"anObject": {
>		"aNumber": 42,
>		"aString": "This is a string",
>		"aBoolean": true,
>		"nothing": null,
>		"anArray": [
>			1,
>			{
>				"name": "value",
>				"anotherName": 12
>			},
>			"something else"
>		]
>	}
>}
>```

---
## YAML
Lo YAML (YAML Ain’t Markup Language) è un linguaggio di serializzazione dei dati pensato per gli umani e viene usato per file di configurazione, per archiviare dati o per scambiare dati

```yaml
user:
	firstName: Jhon
	lastName: Smith
	age: 27
```