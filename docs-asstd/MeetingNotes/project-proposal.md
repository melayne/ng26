# Implementing Multilevel Monte Carlo of the Finite-Element Poisson Problem

## Math 563 – Spring 2026

### Melayne Barker and Neal Kuperman

## Goal

*Implement multigrid methods for the Poisson equation using finite elements in NGSolve and Monte Carlo methods.*

## Expected Outcomes

We will generate code and visualizations for the multigrid finite element solutions to present to the class.

## Materials and Sources

We collected materials from Pablo from his project on the topic in 2024.  


We collected materials from assorted multilevel classes that Melayne has taken including recommended algorithms and code she made previously in Python.  


We will use the NGSolve tutorials and other code examples from Jay and Will’s classes.

### Some of the relevant sources include:

Vassilevski, P. S. (2008). *Multilevel block factorization preconditioners: Matrix-based analysis and algorithms for solving finite element equations*. Springer. [https://doi.org/10.1007/978-0-387-71564-3](https://doi.org/10.1007/978-0-387-71564-3) 

Ovall, J. S. (2024). *Numerical Mathematics*. Society for Industrial and Applied Mathematics. [https://doi.org/10.1137/1.9781611978070](https://doi.org/10.1137/1.9781611978070) 

Giles, M. B. (2015). *Multilevel Monte Carlo Methods*. **Acta Numerica, 24**, 259–328. [https://doi.org/10.1017/S096249291500001X](https://doi.org/10.1017/S096249291500001X) 

## Methods

We will learn and review the concepts of finite elements and the multilevel class to understand the framework of the problem.

### Using NGSolve we will generate:

* $H^1$-conforming piecewise linear finite element space  
  * applied to the Poisson problem with variable coefficients   
  * with a two-level method ($P_1$ and $P_2$)  
  * applied to a multigrid algorithm, preconditioned Conjugate Gradient, using the diagonal $\ell_1$-smoother for the homogeneous problem $A\mathbf{x}=\mathbf{0}$  
  * applied to a multigrid algorithm, preconditioned Conjugate Gradient, using forward Gauss-Seidel for the homogeneous problem $A\mathbf{x}=\mathbf{0}$   
  * applied to a multigrid algorithm, preconditioned Conjugate Gradient, using symmetric Gauss-Seidel for the homogeneous problem $A\mathbf{x}=\mathbf{0}$   
* Run Monte Carlo for sampling for acquiring coefficients  
* Make visualizations of the solution at each step, which will show the smoothing of the highly oscillatory components of the solution (like in the book)

*All project plans are subject to change in response to upcoming meetings with the Professor and researchers.*

