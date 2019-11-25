# Sorting Algorithms

### Bubble Sort

### Selection Sort

### Insertion Sort

### Merge Sort

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
    a[i + 1], a[r] = a[r], a[i+1]
    return i + 1
```

### Heap Sort



