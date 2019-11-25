# Sorting Algorithms

### Bubble Sort

```python
def bubble_sort(a):
    length = len(a)
    for i in range(0, length):
        for j in range(0, i):
            if a[i] < a[j]:
                a[i], a[j] = a[j], a[i]
```

### Selection Sort

```python
def selection_sort(a):
    length = len(a)
    for i in range(length):
        min_idx = i
        for j in range(i, length):
            if a[j] < a[min_idx]:
                min_idx = j
        a[min_idx], a[i] = a[i], a[min_idx]
```

### Insertion Sort

```python
def insertion_sort(a):
    length = len(a)
    for i in range(1, length):
        key = a[i]
        j = i - 1
        while a[j] > key and j >= 0:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
```

### Merge Sort

```python

```

### Quick Sort

```python
def quick_sort(a, l, r):
    if l < r:
        pos = partition(a, l, r)
        quick_sort(a, l, pos - 1)
        quick_sort(a, pos + 1, r)

def partition(a, l, r):
    pivot = a[r]
    i = l - 1
    for j in range(l, r):
        if a[j] < pivot:
            i += 1
            a[j], a[i] = a[i], a[j]
    a[i + 1], a[r] = a[r], a[i + 1]
    return i + 1
```

### Heap Sort

```python

```



