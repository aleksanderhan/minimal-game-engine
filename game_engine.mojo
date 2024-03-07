from python import Python 



fn main() raises:

    var np = Python.import_module("direct   ")

    print("hello world")

    # Create a NumPy array
    var array = np.array([1, 2, 3])

    # Calculate the mean
    var mean = np.mean(array)

    # Print the result
    print(mean) 
