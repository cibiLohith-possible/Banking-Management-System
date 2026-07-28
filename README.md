# 🏦 Banking Management System

A console-based Banking Management System built in Python from the ground up —
covering core language fundamentals through OOP design, custom error handling,
and file persistence. Built task-by-task as a learning project.

## 📋 Topics Covered

| Python Topic | Feature |
|---|---|
| `int`, `Decimal`, `bool` | Balance, Account No, Active Status |
| string | Customer Name, Account Type |
| list | Transactions |
| dictionary | Store accounts |
| comprehension | Filter accounts |
| error handling | Invalid withdrawal, duplicate account |
| file handling | Save/load accounts |
| class & object | Account, Customer, Bank |
| constructor | Initialize account details |
| inheritance | SavingsAccount, CurrentAccount |
| polymorphism | Different interest calculations |
| functions | Deposit, Withdraw, Transfer |
| lambda | Sort accounts |

> **Note on `Decimal`:** balances and monetary amounts use `decimal.Decimal`
> instead of `float`, to avoid binary floating-point rounding errors
> (`0.1 + 0.2 != 0.3`) — not acceptable for financial data.

## 🏗️ Project Structure

```
banking-management-system/
├── exceptions.py     # custom exception classes
├── models.py         # Account, SavingsAccount, CurrentAccount, Customer, Bank
├── main.py           # CLI menu / entry point
├── .gitignore
└── README.md
```
*(accounts.json is generated at runtime and is gitignored - it's user data, not source code)*

## ✅ Features

- Create Savings and Current accounts with type-specific rules
- Deposit, withdraw, and transfer funds with full validation
- Overdraft support on Current accounts
- Minimum-balance protection on Savings accounts
- Interest calculation (polymorphic — differs per account type)
- Transaction history per account, with timestamps
- Filter and sort accounts (by balance, type, name)
- Persistent storage between runs (JSON)
- Custom exceptions instead of generic errors (`InsufficientFundsError`,
  `InactiveAccountError`, `DuplicateAccountError`, etc.)

## 🚀 How to Run

```bash
python main.py
```

## 📅 Progress Log

*Updated as each task is completed — see the full task breakdown in
`docs/roadmap.md`.*

| Day | Task(s) Completed | Notes |
|---|---|---|
| 1 | Task 1 — Account class | Constructor, validation, `__str__` |
| 1 | Task 2 — Deposit & Withdraw | |
| 1 | Task 3 — Custom Exceptions | |
| 2 | Task 4 — Customer class | |
| 2 | Task 5 — Bank class (dict storage) | |
| 2 | Task 6 — Inheritance (Savings/Current) | Switched balance to `Decimal` here |
| |   Task 7 — Polymorphism (interest) | |
| | | |

*(fill in your actual dates as you go — this table is your commit-history story)*

## 🗂️ Task Checklist

- [x] Task 1 — Account class (constructor, `int`/`bool`/`str`/`Decimal`)
- [x] Task 2 — Deposit & Withdraw
- [x] Task 3 — Custom Exceptions
- [x] Task 4 — Customer class
- [x] Task 5 — Bank class (dictionary storage)
- [x] Retrofit — Bank auto-generates account numbers
- [x] Task 6 — Inheritance: SavingsAccount & CurrentAccount
- [x] Retrofit — SavingsAccount enforces `min_balance` on withdraw
- [x] Task 7 — Polymorphism: interest calculation
- [x] Task 8 — Transaction history (with timestamps)
- [x] Task 9 — Transfer between accounts (atomic)
- [x] Task 10 — Filter accounts (list comprehension)
- [x] Task 11 — Sort accounts (lambda)
- [x] Task 12 — Save accounts to file (JSON)
- [x] Task 13 — Load accounts from file (JSON)
- [x] Task 14 — CLI menu + final testing

**Project complete.** ✅

## 🛠️ Built With

- Python 3.x (standard library only)
- `decimal.Decimal` — accurate currency arithmetic
- `json` — data persistence

## 📖 Design Notes

- **Layered design:** `Account`/`Customer` hold domain rules only (no I/O);
  `Bank` owns business rules (account creation, duplicate prevention, transfers);
  the CLI (`main.py`) is the only place user input/output happens.
- **Account numbers are bank-generated**, not user-supplied — mirrors how real
  banking systems avoid collisions and let users pick arbitrary IDs.
- **Transfers are atomic** — if a deposit fails after a withdrawal already
  succeeded, the source account is refunded rather than losing the funds.
