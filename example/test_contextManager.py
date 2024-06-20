from contextlib import contextmanager

@contextmanager
def my_context():
    print("Entering the context!")
    yield "some value"
    print("Exiting the context~~")

with my_context() as value:
    print(f"Inside the context: {value}")
