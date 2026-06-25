Recent Neal work in main branch
Double check dual norm against theory in resources
(1) Single level smoothing demo-ing a toy problem, then GS and
(1b) one to coarse direct solve come back
(2) single v cycle (plotting after a single one)(but doing up to 25)
 of 3 levels (finest, mid, coarse) 
maybe use his v-cycle diagram
"dual norm" converging plots for v-cycles in this part (cycle number vs norm)
(3) Fix tolerance, keeping ncoarse the same and changing how many refinements
demonstrating convergence in # of cycles is independent of mesh resolution
(4) Fix tolerance, keep nfine the same and change what coarse level you go to
demonstrating same concept as three
(5) v-cycle preconditioner being used by PCG

* might be good to show comparison against other things, 
    - cg on on a level size
    - their pcg on that size
    - (5) for same problem

on line 368 solver is defined: needs are listed


