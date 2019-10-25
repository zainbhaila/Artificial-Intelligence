import racetrack as rt
import sample_probs as sp
import sample_heuristics as sh
rt.main(sp.lhook16, "a*", sh.h_walldist, draw=1, verbose=0)
