
## data type

Why are dictionaries faster than lists for key-based lookups?

key-based lookups very fast
- Dictionaries in Python use a hash table internally, which allows them to directly compute the location of the value associated with a given key with an average complexity of O(1). 
- Lists, on the other hand, require a sequential search (O(n)) to find an item.


Which of the following statements is true regarding the performance of Pandas and NumPy? 
- NumPy is faster than Pandas for numerical computations because it operates on homogeneous arrays (all elements have the same data type), which allows better optimization and performance. 
- Pandas, while versatile and feature-rich for data manipulation, operates on heterogeneous data structures like DataFrames, which introduce additional overhead. 
- Both rely on C-based optimizations


## Function
What is a closure in Python?

内部函数记住并访问其外部函数的变量
闭包的三个必要条件
嵌套函数：函数内部定义了另一个函数。
内部函数引用了外部函数的变量。
外部函数返回了内部函数。

```py
def outer(x):
    def inner(y):
        return x + y
    return inner

add_five = outer(5)
print(add_five(3))  # 输出：8

# outer(5) 返回了 inner 函数，但 inner 仍然记住了 x = 5。
# 即使 outer 已经执行完毕，inner 仍然可以访问 x，这就是闭包。


# outer 函数作用域：
# ┌────────────┐
# │ x = 10     │
# └────────────┘
#        ↓
# inner 函数对象：
# ┌────────────────────────────┐
# │ code: return x + y         │
# │ freevars: ['x']            │
# │ __closure__: cell(x=10)    │ ← 绑定了 outer 的局部变量 x
# └────────────────────────────┘
# 为什么闭包能“记住”变量？
# 因为 Python 在创建闭包时，会将外部函数的局部变量封装进一个 cell 对象中，并将其绑定到内部函数的 __closure__ 属性中。这样即使外部函数已经执行完毕，这些变量依然存在于内存中，不会被垃圾回收。
```

What does a decorator do in Python?
- A decorator is a function that takes another function as an argument and modifies its behavior



What is the output of the following code?
nums = [1, 2, 3]
squared = map(lambda x: x**2, nums)
print(list(squared))

The map function 
- applies the lambda function to each element in nums, producing a list of squared numbers
- Lambda functions can only have one expression
- returns an iterator 


Which of the following creates a list of squares of numbers from 0 to 4?
[x**2 for x in range(5)]  a list comprehension


What are *args and **kwargs in Python?
- *args is used to pass a variable number of extra positional arguments into a tuple
- **kwargs is used to pass a variable number of extra keyword arguments into a dictionary








## Object
What is the main difference between @staticmethod and @classmethod in Python?
- @staticmethod does not access any instance or class attributes
- @classmethod can access class attributes.




What is the difference between __str__ and __repr__ in Python?
定义对象的字符串表示形式

| 特性         | `__str__`                            | `__repr__`                                 |
|--------------|--------------------------------------|---------------------------------------------|
| **目标用户** | for end-user output   | used for debuggin      |
| **返回内容** | 简洁、可读                           | 准确、详细，尽可能可复现对象               |
| **调用方式** | `str(obj)` 或 `print(obj)`           | `repr(obj)` 或直接在解释器中输入对象       |
| **默认行为** | 如果未定义，回退到 `__repr__`        | 如果未定义，使用默认的 `<Class at 0x...>` |


How does Python implement method resolution in multiple inheritance?
C3 linearization algorithm

```py
class A: pass
class B(A): pass
class C(A): pass
class D(B, C): pass

print(D.__mro__)
 D 类的 MRO: (D, B, C, A, object)
```
- MRO（Method Resolution Order）是 Python 在多重继承中决定调用哪个类的方法的顺序。
- C3 线性化是一种算法，用来计算 MRO
  - 子类优先于父类
  - 继承顺序一致


What is the time complexity of calling a method in a Python class?


obj.method()

- looks up method in the class hierarchy using the Method Resolution Order (MRO).
  - 第一次访问类的 MRO（比如第一次调用 ClassName.__mro__ 或第一次调用某个方法）时，Python 会使用 C3 线性化算法 来计算 MRO.这个计算过程的时间复杂度与类的继承结构有关，最坏情况下是 O(n²)，其中 n 是类继承图中的节点数（类的数量）。
  - 一旦计算完成，结果会被缓存，后续访问就是 O(1)。


Which statement is true about method overloading in Python?
- 方法重载指的是：在同一个类中，定义多个方法名相同但参数不同（数量或类型不同）的方法。
- Python does not support method overloading natively
  - 第二个 greet 会覆盖第一个，因为 Python 中函数名是变量，后定义的会替代前面的。
```py
class Example:
    def greet(self, name):
        print("Hello", name)

    def greet(self):
        print("Hi there!")
```
- 如何“模拟”方法重载？使用默认参数
```py
class Example:
    def greet(self, name=None):
        if name:
            print("Hello", name)
        else:
            print("Hi there!")
```



## Error
You are integrating a Generative AI model that outputs text. To handle potential network issues or temporary API downtime, which Python construct is essential for robust API client development?
- handling exceptions that might occur during API calls, such as network errors, temporary API downtime, timeouts, or unexpected responses
- preventing your program from crashing/robust
- Try-Except blocks


## Mem

What is the role of the heap in Python’s memory management?




How does Python handle memory management?
- **automatically** manages memory allocation and deallocation using reference counting and 
- a garbage **collector** to reclaim memory occupied by unreachable objects.


What triggers garbage collection in Python?
- when an object is no longer referenced (reference count drops to zero) or 
- when the garbage collector detects circular references (reference count not zero), but unreachable.

What is the main benefit of weak references in Python?
Weak references 
- 是一种引用对象的方式，它不会增加对象的引用计数，因此不会阻止对象被垃圾回收（GC）
- allow objects to be garbage collected even when they are still referenced by a weak reference, helping to save memory.


How can you manually manage memory in Python?
gc module
- enable or disable garbage collection
- manually triggering garbage collection - gc.collect()



What is the main advantage of using __slots__ in a Python class?
- 每个对象默认都有一个 __dict__ 属性，这是一个字典，用来动态存储实例的属性和值
```py
class Person:
    def __init__(self, name):
        self.name = name

p = Person("Alice") 
p.age = 30 # 这个字典的好处是灵活，可以随时添加新属性
print(p.__dict__)  # {'name': 'Alice', 'age': 30}
```
- 缺点是：字典结构本身占用较多内存，尤其是当你创建成千上万个对象时，这种开销就非常明显。
- reduces memory usage 

- 当你在类中定义了 __slots__，Python 会：
  - 限制实例只能拥有 __slots__ 中定义的属性limiting the attributes to a fixed set
  - 不再为每个实例创建 __dict__, 使用更紧凑的内部结构（如数组或描述符）来存储属性preventing the creation of the __dict__ for each instance and storing them in a more compact form.
```py
class Person:
    __slots__ = ['name', 'age']

    def __init__(self, name, age):
        self.name = name
        self.age = age
# 只能设置 name 和 age，不能添加其他属性
# Person 实例不会有 __dict__
```


## Package
What is the purpose of the global keyword in Python?
- allows a variable defined inside a function to be accessed globally



What is the difference between a module and a package in Python?
- A module is a single Python file that contains functions, classes, or variables
- a package is a directory that contains multiple modules and a special __init__.py file.

What is the purpose of the __init__.py file in Python packages?
- signals to Python that the directory is a package could be imported
- include initialization code or make specific modules available when the package is imported.

How does the import statement work in Python?
- searches for the specified module
- executes its code (if it hasn't already been executed)
- loads the module's objects into the current namespace.

## API
When working with large Generative AI models that might take a significant time to process a request, which common API design pattern might be implemented by the GenAI service to handle long-running operations?
- For normal task
  - Synchronous Request-Response
- For long-running tasks
  - Asynchronous - APIs often respond immediately with a job ID
  - Request-Polling - the client then polls a separate status endpoint with that ID until the results are ready
- For the server to notify the client
  - webhooks
- For real-time continuous data
  - WebSocket streaming


Which module is better suited for use in FastAPI applications due to its async support?
- requests - block the thread/sync
- httpx 
  - block the thread/sync
  - non-blocking HTTP calls inside asynchronous routes/async


What is the main purpose of a Python serializer?
- Serialization in Python converts objects into formats like JSON or binary to store or transfer them.
- Deserialization reverses this process to restore the original object. 
- Libraries like pickle and json are commonly used for serialization.



## concurrency

What is the purpose of the Global Interpreter Lock (GIL) in Python?
- CPython 的内存管理不是线程安全的。
  - CPython 使用了 引用计数（reference counting） 来管理内存。
  - 每个对象都有一个引用计数器，记录有多少地方引用了它。
  - 如果多个线程同时修改这个计数器，就可能导致内存错误或崩溃。
CPython 引入了 GIL：在任意时刻，只允许一个线程执行 Python 代码，从而避免了引用计数的并发问题。