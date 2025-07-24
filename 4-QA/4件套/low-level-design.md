
## Design pattern
Which SOLID principle states that a class should have only one reason to change?
- Single Responsibility Principle




Which design pattern is demonstrated by Python’s __new__ method?

The __new__ method can be used to implement the Singleton pattern by ensuring that only one instance of a class is created.

- 数据库连接池,线程池
- 单例模式（Singleton Pattern）- 一个类只能有一个实例，并提供一个全局访问点。
- 在 __new__ 中判断是否已经创建过实例，如果有，就直接返回已有的实例
```py
class Singleton:
    _instance = None  # 用于保存唯一实例

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        # __new__ 必须返回一个类的实例（通常是 super().__new__(cls)）
        # 如果你返回的不是该类的实例，__init__ 就不会被调用
        return cls._instance

    def __init__(self, value):
        self.value = value


a = Singleton(10)
b = Singleton(20)

print(a is b)         # True，说明是同一个对象

```