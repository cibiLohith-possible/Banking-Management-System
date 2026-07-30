"""Core domain classes for the Banking Management System.

Account / SavingsAccount / CurrentAccount / Customer hold data and business
rules only - no input() or print() for user interaction here. Bank owns the
account collection and higher-level operations (create, transfer, save/load).
"""
from decimal import Decimal
from datetime import datetime
import json

from exceptions import (
    InsufficientFundsError,
    InactiveAccountError,
    InvalidAmountError,
    DuplicateAccountError,
    AccountNotFound,
)


# ==================== Task 1 & 2: Account (base class) ====================
class Account:
    def __init__(self, account_number: int, customer_name: str, account_type: str,
                 balance=Decimal("0.0"), is_active=True):
        # Task 1: constructor - int/str/Decimal/bool attributes
        self.account_number = account_number
        self.customer_name = customer_name
        self.account_type = account_type
        if account_type.lower() not in ["savings", "current"]:
            raise ValueError('Enter invalid data in it')
        self.balance = Decimal(str(balance))
        self.is_active = is_active
        self.transactions = []  # Task 8: transaction history list

    def __str__(self):
        # Task 1: readable account summary
        return (f'Account number: {self.account_number}\n'
                f'Customer name: {self.customer_name}\n'
                f'Account type: {self.account_type}\n'
                f'Balance : {self.balance}')

    def deposit(self, amount):
        # Task 2: deposit function + Task 3: custom exceptions + Task 8: log transaction
        if not self.is_active:
            raise InactiveAccountError('Activate your account first')
        amount = Decimal(str(amount))
        if amount <= 0:
            raise InvalidAmountError('Invalid amount Entered')
        self.balance += amount
        self.transactions.append({
            "type": "deposit", "amount": amount,
            "balance_after": self.balance, "timestamp": datetime.now()
        })
        return f'Amount Deposited to your account {amount}, your balance is {self.balance}'

    def withdraw(self, amount):
        # Task 2: withdraw function + Task 3: custom exceptions + Task 8: log transaction
        if not self.is_active:
            raise InactiveAccountError('your account have Deactivated so Activate first')
        amount = Decimal(str(amount))
        if amount <= 0:
            raise InvalidAmountError("Invalid amount Entered")
        if self.balance < amount:
            raise InsufficientFundsError('Insufficient balance for this withdrawal')
        self.balance -= amount
        self.transactions.append({
            "type": "withdraw", "amount": amount,
            "balance_after": self.balance, "timestamp": datetime.now()
        })
        return f"You have withdrawn {amount}, new balance: {self.balance}"

    def calculate_interest(self):
        # Task 7: polymorphism - base case, no interest by default
        return Decimal("0")

    def print_statement(self):
        # Task 8: print full transaction history
        for txn in self.transactions:
            print(f"{txn['timestamp']} | {txn['type'].upper():8} | "
                  f"amount: {txn['amount']:>10} | balance after: {txn['balance_after']}")

    def to_dict(self):
        # Task 12: convert to a JSON-safe dict (Decimal -> str) for saving
        return {"account_number": self.account_number, "customer_name": self.customer_name,
                "account_type": self.account_type, "balance": str(self.balance),
                "is_active": self.is_active}


# ==================== Task 6: SavingsAccount (inheritance) ====================
class SavingsAccount(Account):
    def __init__(self, account_number, customer_name, balance, interest_rate, min_balance):
        # Task 6: inheritance via super().__init__(), plus subclass-only attributes
        super().__init__(account_number, customer_name, "savings", balance)
        self.interest_rate = Decimal(str(interest_rate))
        self.min_balance = Decimal(str(min_balance))

    def withdraw(self, amount):
        # Retrofit: overridden withdraw that enforces min_balance (real-world savings rule)
        amount = Decimal(str(amount))
        if amount <= 0:
            raise InvalidAmountError("Invalid amount Entered")
        if not self.is_active:
            raise InactiveAccountError("Account is deactivated")
        if (self.balance - amount) < self.min_balance:
            raise InsufficientFundsError(
                f"Withdrawal would drop balance below required minimum of {self.min_balance}"
            )
        self.balance -= amount
        self.transactions.append({
            "type": "withdraw", "amount": amount,
            "balance_after": self.balance, "timestamp": datetime.now()
        })
        return f"You have withdrawn {amount}, new balance: {self.balance}"

    def calculate_interest(self):
        # Task 7: polymorphism - savings-specific interest formula
        return self.balance * self.interest_rate / Decimal("100")

    def to_dict(self):
        # Task 12: extend base to_dict() with subclass-only fields
        data = super().to_dict()
        data["interest_rate"] = str(self.interest_rate)
        data["min_balance"] = str(self.min_balance)
        return data


# ==================== Task 6: CurrentAccount (inheritance) ====================
class CurrentAccount(Account):
    def __init__(self, account_number, customer_name, balance, interest_rate=0, overdraft_limit=1000):
        # Task 6: inheritance via super().__init__(), plus subclass-only attributes
        super().__init__(account_number, customer_name, "current", balance)
        self.interest_rate = Decimal(str(interest_rate))
        self.overdraft_limit = Decimal(str(overdraft_limit))

    def withdraw(self, amount):
        # Task 6: overridden withdraw that allows spending into the overdraft limit
        amount = Decimal(str(amount))
        if amount <= 0:
            raise InvalidAmountError("Invalid amount Entered")
        if not self.is_active:
            raise InactiveAccountError("Account is deactivated")
        if amount > self.balance + self.overdraft_limit:
            raise InsufficientFundsError('Exceeds available balance and overdraft limit')
        self.balance -= amount
        self.transactions.append({
            "type": "withdraw", "amount": amount,
            "balance_after": self.balance, "timestamp": datetime.now()
        })
        return f"You have withdrawn {amount}, new balance: {self.balance}"

    def calculate_interest(self):
        # Task 7: polymorphism - current-account interest formula (0 by default)
        return self.balance * self.interest_rate / Decimal("100")

    def to_dict(self):
        # Task 12: extend base to_dict() with subclass-only fields
        data = super().to_dict()
        data["interest_rate"] = str(self.interest_rate)
        data["overdraft_limit"] = str(self.overdraft_limit)
        return data


# ==================== Task 4: Customer ====================
class Customer:
    def __init__(self, customer_id: int, name: str, email_or_phone: str):
        # Task 4: constructor - a customer can hold multiple accounts (list)
        self.customer_id = customer_id
        self.name = name
        self.email_or_phone = email_or_phone
        self.accounts = []

    def add_account(self, account):
        # Task 4: list.append()
        self.accounts.append(account)

    def total_balance(self):
        # Task 4: sum balances across all of this customer's accounts
        total = 0
        for acc in self.accounts:
            total += acc.balance
        return total


# ==================== Task 5: Bank ====================
class Bank:
    def __init__(self):
        # Task 5: dictionary storage {account_number: Account}
        self.accounts = {}
        self._next_account_number = 1001  # Retrofit: bank-generated account numbers

    def _generate_account_number(self):
        # Retrofit: real banks assign account numbers, users don't pick their own
        number = self._next_account_number
        self._next_account_number += 1
        return number

    def create_account(self, customer_name, account_type, balance=Decimal("0.0"), **extra_fields):
        # Task 5 + Retrofit: builds the right subclass and registers it in self.accounts
        account_number = self._generate_account_number()
        account_type = account_type.lower()
        if account_type == "savings":
            account = SavingsAccount(
                account_number, customer_name, balance,
                extra_fields.get("interest_rate", Decimal("0")),
                extra_fields.get("min_balance", Decimal("0")),
            )
        elif account_type == "current":
            account = CurrentAccount(
                account_number, customer_name, balance,
                extra_fields.get("interest_rate", Decimal("0")),
                extra_fields.get("overdraft_limit", Decimal("1000")),
            )
        else:
            raise ValueError("Enter invalid data in it")
        self.accounts[account.account_number] = account
        return account

    def add_account(self, account):
        # Task 5: duplicate-account error handling; also used by Task 13 (loading)
        if account.account_number in self.accounts:
            raise DuplicateAccountError()
        self.accounts[account.account_number] = account

    def get_account(self, account_number):
        # Task 5: lookup + AccountNotFound error handling
        if account_number not in self.accounts:
            raise AccountNotFound(f"Account {account_number} not found")
        return self.accounts[account_number]

    def close_account(self, account_number):
        # Task 5: deactivate rather than delete (keeps transaction history)
        account = self.get_account(account_number)
        account.is_active = False

    def transfer(self, from_acc_no, to_acc_no, amount):
        # Task 9: transfer between accounts, with atomic refund on failure
        from_account = self.get_account(from_acc_no)
        to_account = self.get_account(to_acc_no)
        from_account.withdraw(amount)
        try:
            to_account.deposit(amount)
        except Exception as e:
            from_account.deposit(amount)  # refund - deposit failed after withdraw succeeded
            raise e
        return (f"Transferred {amount} from account {from_account.account_number} "
                f"to account {to_account.account_number}, "
                f"New balances -> {from_account.account_number}: {from_account.balance}, "
                f"{to_account.account_number}: {to_account.balance}")

    def active(self):
        # Task 10: list comprehension - filter active accounts
        return [acc for acc in self.accounts.values() if acc.is_active]

    def balance_value(self, threshold=Decimal("500")):
        # Task 10: list comprehension - filter by balance threshold
        return [acc for acc in self.accounts.values() if acc.balance > threshold]

    def savings_only(self):
        # Task 10: list comprehension - filter by account type (isinstance)
        return [acc for acc in self.accounts.values() if isinstance(acc, SavingsAccount)]

    def sort_balance(self):
        # Task 11: lambda + sorted() - richest first
        return sorted(self.accounts.values(), key=lambda a: a.balance, reverse=True)

    def sort_customer_name(self):
        # Task 11: lambda + sorted() - alphabetical
        return sorted(self.accounts.values(), key=lambda a: a.customer_name)

    def save_to_file(self, filename="accounts.json"):
        # Task 12: file handling - write all accounts as JSON
        data = [account.to_dict() for account in self.accounts.values()]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename="accounts.json"):
        # Task 13: file handling - rebuild accounts (correct subclass + Decimal) from JSON
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            return  # no saved file yet - nothing to load

        for dic in data:
            account_type = dic["account_type"]
            account_number = dic["account_number"]
            customer_name = dic["customer_name"]
            balance = dic["balance"]

            if account_type == "savings":
                account = SavingsAccount(
                    account_number, customer_name, balance,
                    dic["interest_rate"], dic["min_balance"]
                )
            elif account_type == "current":
                account = CurrentAccount(
                    account_number, customer_name, balance,
                    interest_rate=dic["interest_rate"],
                    overdraft_limit=dic["overdraft_limit"]
                )
            else:
                continue

            account.is_active = dic["is_active"]
            self.add_account(account)

        # Retrofit: keep auto-numbering consistent with what was just loaded
        if self.accounts:
            self._next_account_number = max(self.accounts.keys()) + 1
