package cetus.analysis;

import cetus.hir.*;

import java.util.*;

@Deprecated
public class DataFlow {

    public static Set getDefSet(BasicBlock block) {
        return null;
    }

    public static Set<Expression> defines(Traversable root) {
        TreeSet<Expression> set = new TreeSet<Expression>();
        Map<Expression, Expression> map = defSet(root);
        for (Map.Entry<Expression, Expression> entry : map.entrySet()) {
            if (entry.getValue() == null) {
                set.add(entry.getKey());
            }
        }
        return set;
    }

    public static TreeSet<Expression> mayDefine(Traversable root) {
        TreeSet<Expression> set = new TreeSet<Expression>();
        set.addAll(defSet(root).keySet());
        return set;
    }

    /**
    * Determines the set of definitions in the IR subtree starting
    * at the specified root.  The definitions can be general
    * expressions because of pointer arithmetic.
    *
    * @return a map of definitions at and below the root to their
    *   guarding expression, if any
    */
    public static TreeMap<Expression, Expression> defSet(Traversable root) {
        TreeMap<Expression, Expression> var_to_guard =
                new TreeMap<Expression, Expression>();
        defSetInternal(var_to_guard, null, root);
        return var_to_guard;
    }

    private static void addDef(TreeMap<Expression, Expression> var_to_guard,
                               Expression defined, Expression guard) {
        if (var_to_guard.containsKey(defined)) {
            Expression existing_guard = var_to_guard.get(defined);
            if (existing_guard == null || guard == null) {
                var_to_guard.put(defined, guard);
            } else {
                var_to_guard.put(defined,
                        new BinaryExpression(existing_guard.clone(),
                                             BinaryOperator.LOGICAL_OR,
                                             guard.clone()));
            }
        } else {
            var_to_guard.put(defined, guard);
        }
    }

    @SuppressWarnings("unchecked")
    private static void defSetInternal(
            TreeMap<Expression, Expression> var_to_guard,
            Expression guard, Traversable root) {
        if (root instanceof Expression) {
            BreadthFirstIterator iter = new BreadthFirstIterator(root);
            HashSet<Class> of_interest = new HashSet<Class>();
            of_interest.add(AssignmentExpression.class);
            of_interest.add(UnaryExpression.class);
            for (;;) {
                Object o = null;
                try {
                    o = iter.next(of_interest);
                } catch(NoSuchElementException e) {
                    break;
                }
                if (o instanceof AssignmentExpression) {
                    AssignmentExpression expr = (AssignmentExpression)o;
                    /* Only the left-hand side of an AssignmentExpression
                       is a definition.  There may be other nested
                       definitions but, since iter is not set to prune
                       on AssignmentExpressions, they will be found during
                       the rest of the traversal. */
                    addDef(var_to_guard, expr.getLHS(), guard);
                } else {
                    UnaryExpression expr = (UnaryExpression)o;
                    UnaryOperator op = expr.getOperator();
                    /* there are only a few UnaryOperators that create
                       definitions */
                    if (UnaryOperator.hasSideEffects(op)) {
                        addDef(var_to_guard, expr.getExpression(), guard);
                    }
                }
            }
        } else if (root instanceof IfStatement) {
            IfStatement if_stmt = (IfStatement)root;
            Expression condition = if_stmt.getControlExpression().clone();
            if (guard != null) {
                condition = new BinaryExpression(guard.clone(),
                        BinaryOperator.LOGICAL_AND, condition.clone());
            }
            defSetInternal(var_to_guard, condition, if_stmt.getThenStatement());
            if (if_stmt.getElseStatement() != null) {
                Expression inverted_condition = new UnaryExpression(
                        UnaryOperator.LOGICAL_NEGATION,
                        if_stmt.getControlExpression().clone());
                if (guard != null) {
                    inverted_condition = new BinaryExpression(guard.clone(),
                            BinaryOperator.LOGICAL_AND,
                            inverted_condition.clone());
                }
                defSetInternal(var_to_guard, inverted_condition,
                               if_stmt.getElseStatement());
            }
        } else if (root instanceof Loop) {
            Loop loop = (Loop)root;
            Expression condition = loop.getCondition().clone();
            if (guard != null) {
                condition = new BinaryExpression(guard.clone(),
                        BinaryOperator.LOGICAL_AND, condition.clone());
            }
            defSetInternal(var_to_guard, condition, loop.getBody());
        } else {
            FlatIterator iter = new FlatIterator(root);
            while (iter.hasNext()) {
                Object obj = iter.next();
                defSetInternal(var_to_guard, guard, (Traversable)obj);
            }
        }
    }

    public static Set getUseSet(BasicBlock block) {
        return null;
    }

    /**
    * Determines the set of uses in the IR subtree starting
    * at the specified root.  The uses can be general expressions
    * because of pointer arithmetic.
    *
    * @return the set of uses at and below root
    */
    @SuppressWarnings("unchecked")
    public static Set<Expression> getUseSet(Traversable root) {
        /* this is the set that will be returned */
        TreeSet<Expression> set = new TreeSet<Expression>();
        DepthFirstIterator iter = new DepthFirstIterator(root);
        iter.pruneOn(AccessExpression.class);
        iter.pruneOn(AssignmentExpression.class);
        HashSet<Class> of_interest = new HashSet<Class>();
        of_interest.add(AccessExpression.class);
        of_interest.add(ArrayAccess.class);
        of_interest.add(AssignmentExpression.class);
        of_interest.add(IDExpression.class);
        for (;;) {
            Object o = null;
            try {
                o = iter.next(of_interest);
            } catch(NoSuchElementException e) {
                break;
            }
            if (o instanceof AccessExpression) {
                AccessExpression expr = (AccessExpression)o;
                /* The left-hand side of an access expression
                   is read in the case of p->field.  For accesses
                   like p.field, we still consider it to be a use
                   of p because it could be a use in C++ or Java
                   (because p could be a reference) and it doesn't
                   matter for analysis of C (because it will never
                   be written. */
                set.addAll(getUseSet(expr.getLHS()));
                /* The entire expression is also accessed. */
                set.add(expr);
            } else if (o instanceof AssignmentExpression) {
                AssignmentExpression expr = (AssignmentExpression)o;
                /* Recurse on the right-hand side because it is being read. */
                set.addAll(getUseSet(expr.getRHS()));
                /* The left-hand side also may have uses, but unless the
                   assignment is an update like +=, -=, etc. the top-most
                   left-hand side expression is a definition and not a use. */
                /*
                   For example, when expr is "x[i+1] = c" 
                   temp will contain [ i, x, x[i+1] ] because the recursive call
                   to the getUseSet will add ArrayAccess, x[i+1], and two
                   IDExpression, x and i, in the set. And the following
                   if-statement will remove ArrayAccess, x[i+1], 
                   from temp and the temp will finally be [ i, x ]
                */
                Set temp = getUseSet(expr.getLHS());
                if (expr.getOperator() == AssignmentOperator.NORMAL) {
                    temp.remove(expr.getLHS());
                }
                set.addAll(temp);
            } else {
                /* ArrayAccesses and IDExpressions are uses */
                set.add((Expression)o);
            }
        }
        return set;
    }

    public static void partitionScalarsAndArrays(Set<Expression> scalar_set,
            Set<ArrayAccess> array_set, Set<Expression> initial_set) {
        scalar_set.clear();
        array_set.clear();
        for (Expression expr : initial_set) {
            if (expr instanceof ArrayAccess) {
                array_set.add((ArrayAccess)expr);
            } else {
                scalar_set.add(expr);
            }
        }
    }
}
