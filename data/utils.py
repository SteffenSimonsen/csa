# data/src/utils.py
def pretty_print_sample(sample, verbose=False):
    for key, value in sample.items():
        if verbose:
            print(f"{key:15} ({type(value).__name__}) -> {value}")
        else:
            print(f"{key:15} -> {value}")
    print("-" * 50)