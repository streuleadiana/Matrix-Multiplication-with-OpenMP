# Matrix-Multiplication-with-OpenMP
## Problem Description

In this assignment, you will provide several implementations of matrix multiplication algorithms, both **serial** and **parallel** versions, and analyze their performance. This assignment continues the discussion about matrix multiplication started in `Lecture DenseMatrix.pdf`.

## Core Requirements

### 1. Loop Order Permutations

First, explore several options for different implementations of square matrix multiplications, resulting from the possible permutations in the order in which multiplication loops `i`, `j`, `k` are performed.

* Implement matrix multiplication variants corresponding to all **6 permutations of (i,j,k)**:
    * `i-j-k`
    * `i-k-j`
    * `j-i-k`
    * `j-k-i`
    * `k-i-j`
    * `k-j-i`
* For all 6 algorithms, implement a **serial** version and a **parallel** version using OpenMP.
* *Note: The `i-j-k` and `i-k-j` variants have been discussed in the Lecture and their implementation is given in `omp_matrix_mult.c`.*

### 2. Blocked Matrix Multiplication

Implement the **blocked matrix multiplication** algorithm as suggested in `Lecture DenseMatrix.pdf`.

* Implement the blocked algorithm in a **serial** version and a **parallel** version using OpenMP.
* Take into account also the case when the size of the matrix is not evenly divisible by the block size.
* Perform experiments to determine which is a good block size for your computer.

### 3. Constraints and Validation

* **Matrix Size:** For all algorithms, work with square matrices N*N, where size `N` varies between **1000 and 3000**.
* **Data:** The matrix values can be generated randomly.
* **Validation:** For each algorithm version, **validate its result** by implementing an automatic comparison with the result produced by the classical `i-j-k` algorithm version.

### 4. Performance Analysis

* Determine which version has the best **serial time** and which version has the best **parallel time**.
* Present and discuss performance measurements for all algorithms, serial and parallel versions.

## Resources and Additional Reading

* **Lecture:** `Lecture DenseMatrix.pdf`
* **Base Code:** `omp_matrix_mult.c`
* **Reading 1:** `why-loops-do-matter-a-story-of-matrix-multiplication`
* **Reading 2:** `Anatomy of High-Performance Many-Threaded Matrix Multiplication`
