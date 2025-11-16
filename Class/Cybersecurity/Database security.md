---
Class: "[[Cybersecurity]]"
Related:
---
---
## Introduction
There are many reasons why database security has not kept pace with the increased reliance on databases, such as:
- there is a dramatic imbalance between the complexity of modern database management systems (DBMS) and the security technique used to protect these critical systems
- the increasing reliance on cloud technology to host part or all the corporate database
- most enterprise environments consist of a heterogeneous mixture of database platforms, enterprise platforms, and OS platforms, crating an additional complexity hurdle for security personnel
- the typical organization lacks full-time security personnel
- effective database security requires a strategy based on a full understanding of the security vulnerabilities of SQL
- databases have a sophisticated interaction protocol, Structured Query Language (SQL) which is complex

### Databases
Databases are structured collection of data stored for use by one or more applications and contains the relationships between data items and groups of data items. It can sometimes contain sensitive data that needs to be secured

### DBMS architecture
![[Pasted image 20251117001038.png|400]]

---
## SQL injection attacks (SQLi)
SQL injection attack is one of the most prevalent and dangerous network-based security threats and it’s designed to exploit the nature of Web application pages.

It works by sending malicious SQL commands to the database server and the most common attack goal is bulk extraction of data. Depending on the environment, SQL injection can also be exploited to:
- modify or delete data
- execute arbitrary operating system commands
- launch denial-of-service (DoS) attacks

### Typical SQL injection attack
![[Pasted image 20251117001410.png|400]]

>[!example]
>Authentication bypass using [[#Inband attacks|tautologies]]
>##### 1.
>1. query
>	- `$q = "SELECT id FROM users WHERE user='".$user."' AND pass='".$pass."'";`
>2. sent parameters
>	- `$user = "admin";`
>	- `$pass = "' OR '1'='1";`
>3. query that is really executed
>	- `$q = "SELECT id FROM users WHERE user='admin' AND pass='' OR '1'='1'";`
>- if the input were sanitized (e.g., `mysql_escape_string()`):
>	- `$q = "SELECT id FROM users WHERE user='admin' AND pass='\' OR \'\'=\''";`
>
>##### 2.
>Choosing “blindly” the first available user
>- `$pass = "' OR 1=1 # ";`
>→ `$q = "SELECT id FROM users WHERE user='' AND pass='' OR 1=1 # '";`
>- `$user = "' OR user LIKE '%' #";`
>→ `$q = "SELECT id FROM users WHERE user='' OR user LIKE '%' #' AND pass=''";`
>- `$user = "' OR 1 # ";`
>→ `$q = "SELECT id FROM users WHERE user='' OR 1 #' AND pass=''";`
>
>##### 3.
>Choosing a known user
>- `$user = "admin' OR 1 # ";`
>→ `$q = "SELECT id FROM users WHERE user='admin' OR 1 #' AND pass=''";`
>- `$user = "admin' #";`
>→ `$q = "SELECT id FROM users WHERE user='admin' #' AND pass=''";`
>
>##### 4.
>IDS evasion
>- `$pass = "' OR 5>4 OR password='mypass";`
>- `$pass = "' OR 'vulnerability'>'server";`

### Technique
The SQLi attack typically works by prematurely terminating a text string and appending a new command. Because the inserted command may have additional strings appended to it before it is executed the attacker terminates the injected string with a comment mark `--` so that the subsequent text is ignored at execution time.

>[!example] Scenario
>Many web applications require the ability to store structured data (e.g., forum, CMS, e-commerce, blog, …) and use a database. We have a SQLi when it is possible to modify the syntax of the query by altering the application input.
>
>Causes:
>- missing input validation
>- application-generated queries contains user-fed input

#### Attack avenues
- **User input** → attackers injects SQL commands by providing suitable crafted user input
- **Server variables** → attackers can forge the values that are placed in HTTP and network headers and exploit this vulnerability by placing data directly into the headers
- **Second-order injection** → a malicious user could rely on data already present in the system or database to trigger an SQL injection attack, so when the attack occurs, the input that modifies the query to cause an attack does not come from the user, but from within the system itself
- **Cookies** → an attacker could alter cookies such that when the application server builds an SQL query based on the cookie’s content, the structure and function of the query is modified
- **Physical user input** → applying user input that constructs an attack outside the realm of web requests

>[!info] SQLi sinks (where to inject the SQL code)
>- user input
>	- GET/POST parameters
>	- many client-side technologies communicate with the server through GET/POST
>- HTTP headers
>	- every HTTP header must be treated as dangerous
>	- `User-Agent`, `Referer`, … can be maliciously altered
>- cookies
>	- after all, they are just headers and the come from the client…
>- the database itself (second order injection)
>	- the input of the application is stored in the database later, the same input may be fetched from the database and used to build another query

### Inband attacks
We talk about inband attacks when it’s used the same communication channel for injecting SQL code and retrieving results. The retrieved data are presented directly in application Web page

![[Pasted image 20251117002616.png|center|600]]

### Inferential attack
We talk about inferential attack when there is no actual transfer of data, but the attacker is able to reconstruct the information by sending particular requests and observing the resulting behavior of the Website/database server. It includes:
- Illegal/logically incorrect queries → this attack lets an attacker gather important information about the type and structure of the backend database of a Web application: this attack is considered a preliminary, information-gathering step for other attacks
- Blind SQL injection → allows attackers to infer the data present in a database system even when the system is sufficiently secure to not display any erroneous information back to the attacker
- Out-of-Band Attack → data is retrieved using a different channel; this can be used when there are limitations on information retrieval, but outbound connectivity from the database server is lax

### Targets
| Target                     | Description                                                                    |
| -------------------------- | ------------------------------------------------------------------------------ |
| identify injectable params | identify sinks                                                                 |
| database footprinting      | find out which DBMS is in use; made easy by wrong error handling               |
| discover DB schema         | table names, column names, column types, privileges                            |
| data extraction            | dumping anything from the DB                                                   |
| data manipulation          | insert, update, delete data                                                    |
| denial of service          | prevent legitimate user from using the web application (`LOCK`, `DELETE`, ...) |
| authentication bypass      | bypass web application authentication mechanism                                |
| remote command execution   | execution of commands not originally provided by the DBMS                      |

### Ending the query
Terminating the query properly can be cumbersome, in fact, frequently, the problem comes from what follows the integrated user parameter. This SQL segment if part of the query and the malicious input must be crafted to handle it without generating syntax errors.
Usually the parameters include comment symbols like: `#`, `--`, `/*...*/`.

