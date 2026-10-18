# KL Expansion Theory Meeting

## Overview
The meeting focused on understanding the KL expansion and its application to random fields. Neal and Panayot discussed the spectral decomposition of the covariance matrix, the eigenvalue problem, and the rotation invariance of distributions. They explored the use of the exponential covariance kernel and its properties, including the need for the exponent of the eigenvectors. The team also discussed the implementation of the KL expansion in Python, the use of NGsolve for discretizing PDEs, and the importance of maintaining the same distribution during multi-level Monte Carlo simulations. They plan to write up their findings and test the KL expansion with computable examples.

## Action Items
- [] Write up a clear, self-contained description of the KL expansion and covariance sampling in Overleaf, suitable for readers unfamiliar with the topic, and share it with the team.
- [] Meet with Diya on Wednesday to review the breakdown of work, discuss the first steps in generating random fields, and coordinate next steps for her coding tasks.
- [] Update the calendar reservation for the next meeting to Wednesday at 1:30 PM and ensure the invite is sent to the relevant participants.
- [] Write a concise summary of the KL expansion and covariance sampling in Overleaf, using the agreed notation, so it can be reused in the project document.
- [] Implement KL expansion sampling for the chosen covariance kernel, generate random coefficient fields, solve the PDE using NGSolve, and run a single-level Monte Carlo simulation to compute the expected outflow flux, aiming to have this working within the next week or two.
- [] Read the Giles et al. paper on CG start and KL expansion to understand the derivation and behavior of the Matérn kernels and KL modes, and prepare notes on this material.
- [] Implement stochastic PDE sampling (e.g., via SPDE-based methods) following the same pattern as the KL expansion work, so that both sampling approaches can be compared.
- [] Implement a multi-level Monte Carlo method for the PDE problem, building on the single-level implementation, with the goal of having it ready in about one to three weeks.

## Outline
Understanding KF Variance and Covariance Matrix
• Panayot and Neal discuss the theory of KF variance and the availability of Python code.
• Neal mentions starting with a paper from Ready and teaching the KL expansion.
• Panayot and Neal agree to start with the covariance matrix and its spectral decomposition.
• Neal explains the goal of understanding how to break x up into components and the importance of rotation invariance.
Spectral Decomposition and Eigenvalue Problem
• Neal delves into the spectral decomposition and the eigenvalue problem.
• Panayot and Neal discuss the square root of eigenvalues and the orthogonality of eigenvectors.
• Neal explains the rotation invariance of the distribution and the ability to rename the matrix.
• Panayot and Neal agree on the notation and the importance of maintaining the same distribution.
KL Expansion and Rotation Invariance
• Neal explains the KL expansion and the ability to write x as a linear combination of eigenvectors.
• Panayot and Neal discuss the importance of rotation invariance and the ability to rename the matrix.
• Neal emphasizes the importance of understanding the KL expansion and its implications.
• Panayot and Neal agree on the notation and the importance of maintaining the same distribution.
Random Functions and Distributions
• Neal introduces the concept of random functions and distributions.
• Panayot and Neal discuss the importance of understanding the KL expansion and its implications.
• Neal mentions a paper by Hale that discusses the hierarchical implementation of the KL expansion.
• Panayot and Neal agree on the importance of understanding the KL expansion and its applications.
Maternal Covariance Kernels
• Melayne and Panayot discuss the use of maternal covariance kernels and their properties.
• Neal explains the importance of using kernels instead of covariance matrices for certain applications.
• Melayne and Panayot discuss the different types of maternal kernels and their properties.
• Neal emphasizes the importance of choosing the right kernel for the application.
Sampling and Truncation
• Neal discusses the importance of sampling and truncation in the KL expansion.
• Panayot and Neal agree on the importance of understanding the KL expansion and its implications.
• Neal explains the process of sampling and truncation in the KL expansion.
• Panayot and Neal discuss the importance of maintaining the same distribution during sampling and truncation.
Multi-Level Monte Carlo Methods
• Neal discusses the importance of multi-level Monte Carlo methods.
• Panayot and Neal agree on the importance of understanding the KL expansion and its applications.
• Neal explains the process of multi-level Monte Carlo methods and their implications.
• Panayot and Neal discuss the importance of maintaining the same distribution during multi-level Monte Carlo methods.
Implementation and Coding
• Neal discusses the implementation and coding of the KL expansion and multi-level Monte Carlo methods.
• Panayot and Neal agree on the importance of understanding the KL expansion and its applications.
• Neal explains the process of coding and implementing the KL expansion and multi-level Monte Carlo methods.
• Panayot and Neal discuss the importance of maintaining the same distribution during coding and implementation.
Next Steps and Future Plans
• Neal discusses the next steps and future plans for the KL expansion and multi-level Monte Carlo methods.
• Panayot and Neal agree on the importance of understanding the KL expansion and its applications.
• Neal explains the process of next steps and future plans for the KL expansion and multi-level Monte Carlo methods.
• Panayot and Neal discuss the importance of maintaining the same distribution during the next steps and future plans.
Meeting Conclusion and Action Items
• Panayot and Neal conclude the meeting and discuss the action items for the next steps.
• Neal emphasizes the importance of understanding the KL expansion and its applications.
• Panayot and Neal agree on the next steps and future plans for the KL expansion and multi-level Monte Carlo methods.
• Melayne and Panayot discuss the importance of maintaining the same distribution during the next steps and future plans.

Transcript
https://otter.ai/u/jgITmsjG4t0N-wKP2BUlItThgU4
