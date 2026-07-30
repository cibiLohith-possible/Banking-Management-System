"""Custom exceptions for the Banking Management System.

# Task 3: Custom Exceptions - error handling via dedicated exception classes
"""


class InsufficientFundsError(Exception):
    """Raised when a withdrawal or transfer exceeds available funds (including overdraft)."""
    pass


class InactiveAccountError(Exception):
    """Raised when an operation is attempted on a closed/deactivated account."""
    pass


class InvalidAmountError(Exception):
    """Raised when a deposit/withdrawal amount is zero or negative."""
    pass


class DuplicateAccountError(Exception):
    """Raised when attempting to register an account number that already exists."""
    pass


class AccountNotFound(Exception):
    """Raised when looking up an account number that doesn't exist."""
    pass
