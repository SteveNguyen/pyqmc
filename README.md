Here are some code about the Quasi-metric method.
Thi is a very experimental (and messy) code for research purpose.
For more details about the method see: http://dx.doi.org/10.1371/journal.pone.0083411

This package depends on 'graph-tool' http://graph-tool.skewed.de/
Unfortunately it seems that the pip install of this package doesn't work.

Principles of use:
* Define your probabilistic variables.
* Define your transition distribution 
* Build the Quasimetric towards a goal and get the Policy.

You can either provide a function to build the transition probability distribution in order to compute everything offline (see samples/test_qm.py).
Or you can build everything online through successive observations (see samples/test_onlineqm.py).
