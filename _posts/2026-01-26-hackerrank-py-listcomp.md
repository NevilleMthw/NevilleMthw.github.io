---
layout: post
title: "HackerRank Python - List Comprehension"
date: 2026-01-26
comments: false
mathjax: true
excerpt: "My understanding of HackerRank Python List Comprehension problem."
---

# Python - List Comprehension

This is my understanding of HackerRank Python question: [https://www.hackerrank.com/challenges/list-comprehensions/problem](<https://www.hackerrank.com/challenges/list-comprehensions/problem>)

The idea is to provide an array of all possible combinations of [i, j, k] for the dimensions of a cuboid, however the sum of the array should not be equal to n.

So we use a list comprehension to solve this by printing a list and within that list we have all the possible combinations allowed. Another key point is using a lexicographical increasing order. My understanding is basically, in the array if the elements are equal it moves to the next element, for example, [0,0,0], [0,0,1], [0,1,1], [1,0,0], [1,0,1], [1,1,1].

The variable n would be a sort of security guard where if the sum is equal to n, it will be removed.

So I started with a simple approach of understanding the problem first. We start with an example first.

If x,y,z each equal to 1, then we would have two choices for each which would be 0 and 1 since in python 1 would mean 0 and 1 if x = 0, then it would be just a single choice of 0.

Therefore, we have eight total choices/combinations. This is the basic concept of how a robot will check every point and provide the combination, each floor, row and seat. This without the filter, variable n.

Let's put it down:

x = 1
y = 1
z = 1

The permutations: **[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]**

If n = 3, we would filter out any arrays which equal to 3.

Final answer: **[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0]**

So let's start with a nested for loop for easier understanding then we move onto converting to a list comprehension.

```python
# Nested for loop:
for n1 in range(x + 1):
    for n2 in range(y + 1):
        for n3 in range(z + 1):
            if sum([n1, n2, n3]) != n:
                print(n1, n2, n3)
```

Now this works quite well, however, this exercise will require the list comprehension.

```python
permutations = [[n1, n2, n3] for n1 in range(x + 1) for n2 in range(y + 1) for n3 in range(z + 1)
if sum([n1, n2, n3])] != n]
print(permutations)
```

Few questions I had doing this problem:

1. Why do we use range(x + 1)? So I understood that we use range since we need to iterate over an integer but if we don't we would get an error saying int is not iterable. So x+1 is basically stating that we can have 0 and 1, if we do not use +1, we would only get the first value as 0.

2. How to convert to a list comprehension? This was the biggest struggle I had since list comprehension syntax can be confusing, so we do not need to add any colons or print statement since in the list comprehension, it will just collect the variables and display it during print.

3. What is [n1, n2, n3]? IN the list comprehension, we would treat these as one unit. So it cannot be n1, n2, n3 or (n1, n2, n3). These are the variable names we intialise for the rest of the list comprehension.

HackerRank username: @nevillemathew201
