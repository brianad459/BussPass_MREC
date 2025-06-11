num = 1.2
num2 = 12


def factorial(n: int) -> int:
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


fact_result = factorial(num2)


print("Factorial of", num2, "is", fact_result)
print("This is it added:", num + num2)
