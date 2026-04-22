import math

def log_transform(values):
    # Apply log(1 + x) to each value
    result = []
    for x in values:
        result.append(math.log1p(x))
    return result


# Example usage
if __name__ == "__main__":
    values1 = [0, 1, 2, 3]
    output1 = log_transform(values1)
    print([round(x, 4) for x in output1])  # [0.0, 0.6931, 1.0986, 1.3863]

    values2 = [99, 999]
    output2 = log_transform(values2)
    print([round(x, 4) for x in output2])  # [4.6052, 6.9078]