from formula import *

# Define common predicates
C = Predicate("C", 1)
E = Predicate("E", 2)
D = Predicate("D", 1)
P = Predicate("P", 1)

# A collection of benchmark formulas
BENCHMARKS = {
    # "exists_c_node": Exists(x, Atom(C, (x,))),
    # "forall_c_node": ForAll(x, Atom(C, (x,))),
    # "exists_edge": Exists(x, Exists(y, Atom(E, (x, y)))),
    # "no_edges": Not(Exists(x, Exists(y, Atom(E, (x, y))))),
    # "exists_self_loop": Exists(x, Atom(E, (x, x))),
    # "no_self_loops": ForAll(x, Not(Atom(E, (x, x)))),
    # "forall_exists_edge": ForAll(x, Exists(y, Atom(E, (x, y)))),
    # "exists_forall_edge": Exists(x, ForAll(y, Atom(E, (x, y)))),
    # "c_nodes_have_loops": ForAll(x, Or(Not(Atom(C, (x,))), Atom(E, (x, x)))),
#     "uncolored_nodes_are_isolated": ForAll(x, ForAll(y, Or(Atom(C, (x,)), Not(Atom(E, (x, y)))))),
#     "c_nodes_to_all_d_and_none_not_p": ForAll(x, Or(
#     Not(Atom(C, (x,))),
#     And(
#         ForAll(y, Implies(Atom(D, (y,)), Atom(E, (x, y)))),
#         ForAll(y, Implies(Not(Atom(P, (y,))), Not(Atom(E, (x, y)))))
#     )
# )),
"uncolored_to_colored_only": ForAll(x, Or(
    Atom(C, (x,)),
    ForAll(y, Implies(Atom(E, (x, y)), Atom(C, (y,))))
))
}
