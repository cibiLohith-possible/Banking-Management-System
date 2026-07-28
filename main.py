"""Command-line entry point for the Banking Management System.

# Task 14: CLI Menu + Final Testing - ties every earlier task together
"""
from decimal import Decimal, InvalidOperation

from models import Bank
from exceptions import (
    InsufficientFundsError,
    InactiveAccountError,
    InvalidAmountError,
    AccountNotFound,
)


def print_menu():
    print("\n--- Banking Menu ---")
    print("1. Create Account")
    print("2. Deposit")
    print("3. Withdraw")
    print("4. Transfer")
    print("5. View All Accounts")
    print("6. Filter Accounts")
    print("7. Sort Accounts")
    print("8. Save & Exit")


def create_account(bank):
    # Task 14 -> uses Task 5's Bank.create_account()
    name = input("Customer name: ").strip()
    acc_type = input("Account type (savings/current): ").strip().lower()
    try:
        balance = Decimal(input("Starting balance: ").strip() or "0")
        if acc_type == "savings":
            rate = Decimal(input("Interest rate (e.g. 4.5 for 4.5%): ").strip() or "0")
            min_bal = Decimal(input("Minimum balance: ").strip() or "0")
            account = bank.create_account(name, "savings", balance=balance,
                                           interest_rate=rate, min_balance=min_bal)
        elif acc_type == "current":
            overdraft = Decimal(input("Overdraft limit: ").strip() or "1000")
            account = bank.create_account(name, "current", balance=balance,
                                           overdraft_limit=overdraft)
        else:
            print("Unknown account type. Must be 'savings' or 'current'.")
            return
        print(f"Account created! Account number: {account.account_number}")
    except (InvalidOperation, ValueError):
        print("Invalid number entered - account not created.")


def deposit(bank):
    # Task 14 -> uses Task 2's Account.deposit()
    try:
        acc_no = int(input("Account number: "))
        amount = Decimal(input("Amount to deposit: "))
        account = bank.get_account(acc_no)
        print(account.deposit(amount))
    except AccountNotFound as e:
        print("Error:", e)
    except (InvalidAmountError, InactiveAccountError) as e:
        print("Error:", e)
    except (InvalidOperation, ValueError):
        print("Invalid input - please enter valid numbers.")


def withdraw(bank):
    # Task 14 -> uses Task 2/6's Account.withdraw() (polymorphic: base/savings/current)
    try:
        acc_no = int(input("Account number: "))
        amount = Decimal(input("Amount to withdraw: "))
        account = bank.get_account(acc_no)
        print(account.withdraw(amount))
    except AccountNotFound as e:
        print("Error:", e)
    except (InsufficientFundsError, InvalidAmountError, InactiveAccountError) as e:
        print("Error:", e)
    except (InvalidOperation, ValueError):
        print("Invalid input - please enter valid numbers.")


def transfer(bank):
    # Task 14 -> uses Task 9's Bank.transfer()
    try:
        from_no = int(input("From account number: "))
        to_no = int(input("To account number: "))
        amount = Decimal(input("Amount to transfer: "))
        print(bank.transfer(from_no, to_no, amount))
    except AccountNotFound as e:
        print("Error:", e)
    except (InsufficientFundsError, InvalidAmountError, InactiveAccountError) as e:
        print("Error:", e)
    except (InvalidOperation, ValueError):
        print("Invalid input - please enter valid numbers.")


def view_all(bank):
    # Task 14 -> uses Task 1's Account.__str__()
    if not bank.accounts:
        print("No accounts yet.")
        return
    for account in bank.accounts.values():
        print(account)
        print("-" * 30)


def filter_accounts(bank):
    # Task 14 -> uses Task 10's Bank.active() / balance_value() / savings_only()
    print("1. Active accounts   2. Balance above threshold   3. Savings only")
    choice = input("Choose filter: ").strip()
    if choice == "1":
        results = bank.active()
    elif choice == "2":
        try:
            threshold = Decimal(input("Threshold balance: "))
        except (InvalidOperation, ValueError):
            print("Invalid number.")
            return
        results = bank.balance_value(threshold)
    elif choice == "3":
        results = bank.savings_only()
    else:
        print("Invalid filter choice.")
        return

    if not results:
        print("No matching accounts.")
    for account in results:
        print(account)
        print("-" * 30)


def sort_accounts(bank):
    # Task 14 -> uses Task 11's Bank.sort_balance() / sort_customer_name()
    print("1. By balance (highest first)   2. By customer name (A-Z)")
    choice = input("Choose sort: ").strip()
    if choice == "1":
        results = bank.sort_balance()
    elif choice == "2":
        results = bank.sort_customer_name()
    else:
        print("Invalid sort choice.")
        return
    for account in results:
        print(account)
        print("-" * 30)


def main():
    bank = Bank()
    bank.load_from_file()  # Task 13 -> restore any previously saved accounts on startup

    while True:
        print_menu()
        choice = input("Choose an option: ").strip()

        if choice == "1":
            create_account(bank)
        elif choice == "2":
            deposit(bank)
        elif choice == "3":
            withdraw(bank)
        elif choice == "4":
            transfer(bank)
        elif choice == "5":
            view_all(bank)
        elif choice == "6":
            filter_accounts(bank)
        elif choice == "7":
            sort_accounts(bank)
        elif choice == "8":
            bank.save_to_file()  # Task 12 -> save before exiting
            print("Saved. Goodbye!")
            break
        else:
            print("Invalid option, try again.")


if __name__ == "__main__":
    main()
