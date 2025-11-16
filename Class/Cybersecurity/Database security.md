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