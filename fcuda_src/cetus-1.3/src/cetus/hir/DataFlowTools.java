package cetus.hir;

import cetus.analysis.RangeDomain;
import cetus.analysis.Section;

import java.util.*;

public final class DataFlowTools {

    private DataFlowTools() {
    }

    /**
    * Returns a set of used expressions in the traversable object.
    *
    * @param t the traversable object.
    * @return the set of used expressions.
    */
    /* JAS
     * Extended original getUseSet function with Class-type parameter for prunning search space
     */
    public static Set<Expression> getUseSet(Traversable t) {
     return getUseSet(t, null); 
    }

    public static Set<Expression> getUseSet(Traversable t, Class<? extends Traversable> c) {
        TreeSet<Expression> ret = new TreeSet<Expression>();
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        // Handle these expressions specially.
        iter.pruneOn(AccessExpression.class);
        iter.pruneOn(ArrayAccess.class);
        iter.pruneOn(AssignmentExpression.class);

	// JAS: filtering search space
	if(c != null)
	    iter.pruneOn(c);

        while (iter.hasNext()) {
            Expression o = iter.next();
            if (o instanceof AccessExpression) {
                AccessExpression ae = (AccessExpression)o;
                DFIterator<ArrayAccess> ae_iter =
                        new DFIterator<ArrayAccess>(ae, ArrayAccess.class);
                ae_iter.pruneOn(ArrayAccess.class);
                // Catches array subscripts in the access expression.
                while (ae_iter.hasNext()) {
                    ArrayAccess aa = ae_iter.next();
                    Set<Expression> aa_use = getUseSet(aa);
                    aa_use.remove(aa);
                    ret.addAll(aa_use);
                }
                ret.add(ae);
            } else if (o instanceof ArrayAccess) {
                ArrayAccess aa = (ArrayAccess)o;
                for (int i = 0; i < aa.getNumIndices(); ++i) {
                    ret.addAll(getUseSet(aa.getIndex(i)));
                }
                ret.add(aa);
            } else if (o instanceof AssignmentExpression) {
                AssignmentExpression ae = (AssignmentExpression)o;
                ret.addAll(getUseSet(ae.getRHS()));
                Set<Expression> lhs_use = getUseSet(ae.getLHS());
                // Other cases should include the lhs in the used set. (+=,...)
                if (ae.getOperator() == AssignmentOperator.NORMAL) {
                    lhs_use.remove(ae.getLHS());
                }
                ret.addAll(lhs_use);
            } else if (o instanceof Identifier) {
                Identifier id = (Identifier)o;
                if (id.getSymbol() instanceof Procedure ||
                    id.getSymbol() instanceof ProcedureDeclarator) {
                    ;
                } else {
                    ret.add(id);
                }
            }
        }
        return ret;
    }

    private static void
            add2Map(Map<Expression, Set<Integer>> map, Expression expr) {
        if (map.containsKey(expr)) {
            Set<Integer> set = map.get(expr);
            set.add(new Integer(System.identityHashCode(expr)));
        } else {
            Set<Integer> set = new HashSet<Integer>();
            set.add(new Integer(System.identityHashCode(expr)));
            map.put(expr, set);
        }
    }

    /**
    * add contents of new_map to orig_map
    */
    public static void mergeSymbolMaps(Map<Symbol, Set<Integer>> orig_map,
                                       Map<Symbol, Set<Integer>> new_map) {
        for (Symbol new_sym : new_map.keySet()) {
            if (orig_map.containsKey(new_sym)) {
                Set<Integer> set = orig_map.get(new_sym);
                set.addAll(new_map.get(new_sym));
            } else {
                Set<Integer> set = new HashSet<Integer>();
                set.addAll(new_map.get(new_sym));
                orig_map.put(new_sym, set);
            }
        }
    }

    /**
    * add contents of new_map to orig_map
    */
    public static void mergeMaps(Map<Expression, Set<Integer>> orig_map,
                                 Map<Expression, Set<Integer>> new_map) {
        for (Expression new_expr : new_map.keySet()) {
            if (orig_map.containsKey(new_expr)) {
                Set<Integer> set = orig_map.get(new_expr);
                set.addAll(new_map.get(new_expr));
            } else {
                Set<Integer> set = new HashSet<Integer>();
                set.addAll(new_map.get(new_expr));
                orig_map.put(new_expr, set);
            }
        }
    }

    public static Map<Symbol, Set<Integer>>
            convertExprMap2SymbolMap(Map<Expression, Set<Integer>> imap) {
        Map<Symbol, Set<Integer>> omap = new HashMap<Symbol, Set<Integer>>();
        for (Expression expr : imap.keySet()) {
            Set<Integer> iset = imap.get(expr);
            Symbol sym = SymbolTools.getSymbolOf(expr);
            if (omap.containsKey(sym)) {
                Set<Integer> orig_set = omap.get(sym);
                orig_set.addAll(iset);
            } else {
                Set<Integer> oset = new HashSet<Integer>();
                oset.addAll(iset);
                omap.put(SymbolTools.getSymbolOf(expr), oset);
            }
        }
        return omap;
    }

    public static Map<Symbol, Set<Integer>> getUseSymbolMap(Traversable t) {
        return convertExprMap2SymbolMap(getUseMap(t));
    }

    public static Map<Symbol, Set<Integer>> getDefSymbolMap(Traversable t) {
        return convertExprMap2SymbolMap(getDefMap(t));
    }

    /**
    * Returns a map from a variable to its section
    */
    public static Section.MAP getUseSectionMap(
            Expression e, RangeDomain rd, Set<Symbol> def_vars) {
        Section.MAP ret = new Section.MAP();
        Expression expr = rd.substituteForward(e);
        if (expr instanceof ArrayAccess) {
            ArrayAccess aa = (ArrayAccess)expr;
            Symbol var = SymbolTools.getSymbolOf(aa.getArrayName());
            Section new_section = new Section(aa);
            new_section.expandMay(rd, def_vars);
            ret.put(SymbolTools.getSymbolOf(aa.getArrayName()), new_section);
        } else if (expr instanceof AccessExpression) {
            Set use_set = getUseSet(expr);
            if (use_set.size() == 1) {
                /*
                AccessSymbol var = (AccessSymbol)SymbolTools.getSymbolOf(expr);
                ret.put( var.get(0), new Section(-1) );
                */
                Symbol var = SymbolTools.getSymbolOf(expr);
                if (var instanceof PseudoSymbol) {
                    var = ((PseudoSymbol)var).getIRSymbol();
                }
                ret.put(var, new Section(-1));
            }
        } else {
            Symbol var = SymbolTools.getSymbolOf(expr);
            // var == null means it is not variable type e.g.) *a = 0;
            if (var != null) {
                ret.put(var, new Section(-1));
            } else {
                /////////////////////////////////////////////////////////
                // [Added by Seyong Lee]                               //
                // If forward-substituted expression, expr, is  not a  //
                // variable type, use original expression, e, instead. //
                /////////////////////////////////////////////////////////
                if (e instanceof ArrayAccess) {
                    ArrayAccess aa = (ArrayAccess)e;
                    Symbol var2 = SymbolTools.getSymbolOf(aa.getArrayName());
                    Section new_section = new Section(aa);
                    new_section.expandMay(rd, def_vars);
                    ret.put(SymbolTools.getSymbolOf(
                            aa.getArrayName()), new_section);
                } else if (e instanceof AccessExpression) {
                    Set use_set = getUseSet(e);
                    if (use_set.size() == 1) {
                        /*
                        AccessSymbol var2 =
                                (AccessSymbol)SymbolTools.getSymbolOf(e);
                        ret.put( var2.get(0), new Section(-1) );
                        */
                        Symbol var2 = SymbolTools.getSymbolOf(e);
                        if (var2 instanceof PseudoSymbol) {
                            var2 = ((PseudoSymbol)var2).getIRSymbol();
                        }
                        ret.put(var2, new Section(-1));
                    }
                } else {
                    Symbol var2 = SymbolTools.getSymbolOf(e);
                    // var2 == null means it is not variable type
                    // e.g.) *a = 0;
                    if (var2 != null) {
                        ret.put(var2, new Section(-1));
                    }
                }

            }
        }
        ret.clean();            // cleans up empty Sections
        return ret;
    }

    /**
    * Returns a map from a variable to its section
    */
    public static Section.MAP getDefSectionMap(
            Expression e, RangeDomain rd, Set<Symbol> def_vars) {
        Section.MAP ret = new Section.MAP();
        Expression expr = rd.substituteForward(e);
        if (expr instanceof ArrayAccess) {
            ArrayAccess aa = (ArrayAccess)expr;
            Symbol var = SymbolTools.getSymbolOf(aa.getArrayName());
            Section new_section = new Section(aa);
            new_section.expandMay(rd, def_vars);
            ret.put(SymbolTools.getSymbolOf(aa.getArrayName()), new_section);
        } else {
            Symbol var = SymbolTools.getSymbolOf(expr);
            // var == null means it is not variable type e.g.) *a = 0;
            if (var != null) {
                ret.put(var, new Section(-1));
            } else {
                /////////////////////////////////////////////////////////
                // [Added by Seyong Lee]                               //
                // If forward-substituted expression, expr, is  not a  //
                // variable type, use original expression, e, instead. //
                /////////////////////////////////////////////////////////
                if (e instanceof ArrayAccess) {
                    ArrayAccess aa = (ArrayAccess)e;
                    Symbol var2 = SymbolTools.getSymbolOf(aa.getArrayName());
                    Section new_section = new Section(aa);
                    new_section.expandMay(rd, def_vars);
                    ret.put(SymbolTools.getSymbolOf(
                            aa.getArrayName()), new_section);
                } else {
                    Symbol var2 = SymbolTools.getSymbolOf(e);
                    // var2 == null means it is not variable type e.g.) *a = 0;
                    if (var2 != null) {
                        ret.put(var2, new Section(-1));
                    }
                }
            }
        }
        ret.clean();            // cleans up empty Sections
        return ret;
    }

    /**
    * Returns a set of used expressions with their unique hashcodes in the 
    * traversable object.
    *
    * @param t the traversable object.
    * @return the set of used expressions.
    */
    public static Map<Expression, Set<Integer>> getUseMap(Traversable t) {
        Map<Expression, Set<Integer>> ret =
                new TreeMap<Expression, Set<Integer>>();
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        // Handle these expressions specially.
        iter.pruneOn(AccessExpression.class);
        iter.pruneOn(ArrayAccess.class);
        iter.pruneOn(AssignmentExpression.class);
        while (iter.hasNext()) {
            Expression o = iter.next();
            if (o instanceof AccessExpression) {
                AccessExpression ae = (AccessExpression)o;
                DFIterator<ArrayAccess> ae_iter =
                        new DFIterator<ArrayAccess>(ae, ArrayAccess.class);
                ae_iter.pruneOn(ArrayAccess.class);
                // Catches array subscripts in the access expression.
                while (ae_iter.hasNext()) {
                    ArrayAccess aa = ae_iter.next();
                    Map<Expression, Set<Integer>> aa_use = getUseMap(aa);
                    aa_use.remove(aa);
                    mergeMaps(ret, aa_use);
                }
                add2Map(ret, ae);
            } else if (o instanceof ArrayAccess) {
                ArrayAccess aa = (ArrayAccess)o;
                for (int i = 0; i < aa.getNumIndices(); ++i) {
                    mergeMaps(ret, getUseMap(aa.getIndex(i)));
                }
                add2Map(ret, aa);
            } else if (o instanceof AssignmentExpression) {
                AssignmentExpression ae = (AssignmentExpression)o;
                mergeMaps(ret, getUseMap(ae.getRHS()));
                // lhs_use is {A[i], i} for both {A[i]=...;} and
                // {A[i]+=...} cases
                Map<Expression, Set<Integer>> lhs_use = getUseMap(ae.getLHS());
                // Other cases should include the lhs in the used set. (+=,...)
                // now, lhs_use is {i} for both {A[i]=...;} and
                // {A[i]+=...} cases
                if (ae.getOperator() == AssignmentOperator.NORMAL) {
                    lhs_use.remove(ae.getLHS());
                }
                mergeMaps(ret, lhs_use);
            } else if (o instanceof Identifier) {
                Identifier id = (Identifier)o;
                if (id.getSymbol() instanceof Procedure ||
                    id.getSymbol() instanceof ProcedureDeclarator) {
                    ;
                } else {
                    add2Map(ret, id);
                }
            }
        }
        return ret;
    }

    public static void
            displayMap(Map<Expression, Set<Integer>> imap, String name) {
        int key_cnt = 0;
        for (Expression expr : imap.keySet()) {
            System.out.print(name+(++key_cnt)+" : "+expr.toString()+" = {");
            int val_cnt = 0;
            for (Integer hashcode : imap.get(expr)) {
                if (val_cnt++ == 0) {
                    System.out.print(hashcode.toString());
                } else {
                    System.out.print(", " + hashcode.toString());
                }
            }
            System.out.println("}");
        }
    }

    /**
    * Returns a set of used expressions in the traversable object.
    *
    * @param t the traversable object.
    * @return the set of used expressions.
    */
    private static Set<Expression> getFlatUseSet(Traversable t) {
        TreeSet<Expression> ret = new TreeSet<Expression>();
        FlatIterator iter = new FlatIterator(t);
        // Handle these expressions specially.
/*
        iter.pruneOn(AccessExpression.class);
        iter.pruneOn(ArrayAccess.class);
        iter.pruneOn(AssignmentExpression.class);
*/
        while (iter.hasNext()) {
            Object o = iter.next();
            if (o instanceof AccessExpression) {
                AccessExpression ae = (AccessExpression)o;
                DepthFirstIterator ae_iter = new DepthFirstIterator(ae);
/*
                iter.pruneOn(ArrayAccess.class);
*/
                // Catches array subscripts in the access expression.
                while (ae_iter.hasNext()) {
                    Object oo = ae_iter.next();
                    if (oo instanceof ArrayAccess) {
                        ArrayAccess aa = (ArrayAccess)oo;
                        Set<Expression> aa_use = getUseSet(aa);
                        aa_use.remove(aa);
                        ret.addAll(aa_use);
                    }
                }
                ret.add(ae);
            } else if (o instanceof ArrayAccess) {
                ArrayAccess aa = (ArrayAccess)o;
                for (int i = 0; i < aa.getNumIndices(); ++i) {
                    ret.addAll(getUseSet(aa.getIndex(i)));
                }
                ret.add(aa);
            } else if (o instanceof AssignmentExpression) {
                AssignmentExpression ae = (AssignmentExpression)o;
                ret.addAll(getUseSet(ae.getRHS()));
                Set<Expression> lhs_use = getUseSet(ae.getLHS());
                // Other cases should include the lhs in the used set. (+=,...)
                if (ae.getOperator() == AssignmentOperator.NORMAL) {
                    lhs_use.remove(ae.getLHS());
                }
                ret.addAll(lhs_use);
            } else if (o instanceof Identifier) {
                Identifier id = (Identifier)o;
                if (id.getSymbol() instanceof Procedure ||
                    id.getSymbol() instanceof ProcedureDeclarator) {
                    ;
                } else {
                    ret.add(id);
                }
            }
        }
        return ret;
    }

    /**
    * Returns a set of defined expressions in the traversable object.
    * 
    * @param t the traversable object.
    * @return the set of defined expressions.
    */
    public static Set<Expression> getDefSet(Traversable t) {
        Set<Expression> ret = new TreeSet<Expression>();
        if (t == null) {
            return ret;
        }
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        while (iter.hasNext()) {
            Expression o = iter.next();
            if (o instanceof AssignmentExpression) {
                ret.add(((AssignmentExpression)o).getLHS());
            } else if (o instanceof UnaryExpression) {
                UnaryExpression ue = (UnaryExpression)o;
                UnaryOperator uop = ue.getOperator();
                if (uop == UnaryOperator.POST_DECREMENT ||
                    uop == UnaryOperator.POST_INCREMENT ||
                    uop == UnaryOperator.PRE_DECREMENT ||
                    uop == UnaryOperator.PRE_INCREMENT) {
                    ret.add(ue.getExpression());
                }
            }
        }
        return ret;
    }

    /**
    * Returns a list of defined expressions in the traversable object.
    * 
    * @param t the traversable object.
    * @return the list of defined expressions.
    */
    public static List<Expression> getDefList(Traversable t) {
        List<Expression> ret = new ArrayList<Expression>();
        if (t == null) {
            return ret;
        }
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        while (iter.hasNext()) {
            Expression o = iter.next();
            if (o instanceof AssignmentExpression) {
                ret.add(((AssignmentExpression)o).getLHS());
            } else if (o instanceof UnaryExpression) {
                UnaryExpression ue = (UnaryExpression)o;
                UnaryOperator uop = ue.getOperator();
                if (uop == UnaryOperator.POST_DECREMENT ||
                    uop == UnaryOperator.POST_INCREMENT ||
                    uop == UnaryOperator.PRE_DECREMENT ||
                    uop == UnaryOperator.PRE_INCREMENT) {
                    ret.add(ue.getExpression());
                }
            }
        }
        return ret;
    }

    /**
    * Returns a set of defined expressions in the traversable object.
    * 
    * @param t the traversable object.
    * @return the set of defined expressions.
    */
    public static Map<Expression, Set<Integer>> getDefMap(Traversable t) {
        TreeMap<Expression, Set<Integer>> ret =
                new TreeMap<Expression, Set<Integer>>();
        if (t == null) {
            return ret;
        }
        // Add increment/decrement operator in search list.
        Set<String> unary_def = new HashSet<String>();
        unary_def.add("--");
        unary_def.add("++");
        DepthFirstIterator iter = new DepthFirstIterator(t);
        while (iter.hasNext()) {
            Object o = iter.next();
            // Expression being modified
            if (o instanceof AssignmentExpression) {
                Expression expr = ((AssignmentExpression)o).getLHS();
                add2Map(ret, expr);
            } else if (o instanceof UnaryExpression) {
                UnaryExpression ue = (UnaryExpression)o;
                if (unary_def.contains(ue.getOperator().toString())) {
                    Expression expr = ue.getExpression();
                    add2Map(ret, expr);
                }
            }
        }
        return ret;
    }

    /**
    * Returns a set of defined Section expressions in the traversable object.
    * 
    * @param t the traversable object.
    * @param range_map the range map of the procedure that contains t
    * @return the set of defined expressions.
    */
    public static Section.MAP getDefSectionMap(Traversable t, Map range_map,
            RangeDomain unioned_rd, Set<Symbol> def_vars) {
        Section.MAP map = new Section.MAP();
        Set<String> unary_def = new HashSet<String>();
        unary_def.add("--");
        unary_def.add("++");
        DepthFirstIterator iter = new DepthFirstIterator(t);
        while (iter.hasNext()) {
            Object o = iter.next();
            Expression def_expr = null;
            if (o instanceof AssignmentExpression) {
                def_expr = ((AssignmentExpression)o).getLHS();
            } else if (o instanceof UnaryExpression) {
                UnaryExpression ue = (UnaryExpression)o;
                if (unary_def.contains(ue.getOperator().toString())) {
                    def_expr = ue.getExpression();
                }
            }
            if (def_expr != null) {
                if (def_expr instanceof ArrayAccess) { // A[i][j], p->q[k], etc
                    ArrayAccess aa_expr = (ArrayAccess)def_expr;
                    Symbol aa_symbol = SymbolTools.getSymbolOf(aa_expr);
                    Section new_section = new Section(aa_expr);
                    Statement stmt = aa_expr.getStatement();
                    RangeDomain rd = (RangeDomain)range_map.get(stmt);
                    // expand symbolic varaibles in subscript expression
                    // specified in def_vars
                    new_section.expandMay(rd, def_vars);
                    // if section_map already contains a (aa_symbol, Section)
                    // pair
                    if (map.keySet().contains(aa_symbol)) {
                        Section old_section = map.get(aa_symbol);
                        if (old_section.getDimension() !=
                                aa_expr.getNumIndices()) {
                            Tools.exit(
                                    "ERROR: array re-shaping is not supported:"
                                    + aa_expr.toString());
                        } else {
                            PrintTools.println("old_section before merge: " +
                                    old_section.toString(), 4);
                            PrintTools.println("new_section before merge: " +
                                    new_section.toString(), 4);
                            // merge two Sections, the previous and the current,
                            // to find out the maximum range of access ranges
                            // for a given ArrayAccess
                            new_section =
                                new_section.unionWith(old_section, unioned_rd);
                            PrintTools.println("old_section after merge: " +
                                    old_section.toString(), 4);
                            PrintTools.println("new_section after merge: " +
                                    new_section.toString(), 4);
                            // remove old_section
                            map.remove(aa_symbol);
                            // insert new_section
                            map.put(aa_symbol, new_section);
                        }
                    } else {
                        map.put(aa_symbol, new_section);  // insert new_section
                        PrintTools.println("Section inserted: " +
                               new_section.toString(), 3);
                    }
                } else {
                    Symbol scalar_symbol = SymbolTools.getSymbolOf(def_expr);
                    map.put(scalar_symbol, new Section(-1));
                }
            }
        }
        return map;
    }

    /**
    * Returns a set of defined expressions in the traversable object.
    * 
    * @param t the traversable object.
    * @return the set of defined expressions.
    */
    private static Set<Expression> getFlatDefSet(Traversable t) {
        Set<Expression> ret = new TreeSet<Expression>();
        if (t == null) {
            return ret;
        }
        // Add increment/decrement operator in search list.
        Set<String> unary_def = new HashSet<String>();
        unary_def.add("--");
        unary_def.add("++");
        FlatIterator iter = new FlatIterator(t);
        while (iter.hasNext()) {
            Object o = iter.next();
            // Expression being modified
            if (o instanceof AssignmentExpression) {
                ret.add(((AssignmentExpression)o).getLHS());
            } else if (o instanceof UnaryExpression) {
                UnaryExpression ue = (UnaryExpression)o;
                if (unary_def.contains(ue.getOperator().toString())) {
                    ret.add(ue.getExpression());
                }
            }
        }
        return ret;
    }

    /**
    * Returns a set of defined symbols from the traversable object.
    *
    * @param t the traversable object.
    * @return the set of defined symbols.
    */
    public static Set<Symbol> getDefSymbol(Traversable t) {
        Set<Symbol> ret = new LinkedHashSet<Symbol> ();
        for (Expression e : getDefList(t)) {
            Symbol symbol = SymbolTools.getSymbolOf(e);
            if (symbol != null) {
                ret.add(symbol);
            }
        }
        return ret;
    }

    /**
    * Returns a set of defined symbols from the traversable object.
    *
    * @param t the traversable object.
    * @return the set of defined symbols.
    */
    private static Set<Symbol> getFlatDefSymbol(Traversable t) {
        Set<Symbol> ret = new HashSet<Symbol>();
        for (Expression e : getFlatDefSet(t)) {
            Symbol symbol = SymbolTools.getSymbolOf(e);
            if (symbol != null) {
                ret.add(symbol);
            }
        }
        return ret;
    }

    /**
    * Returns a set of used symbols from the traversable object.
    *
    * @param t the traversable object.
    * @return the set of used symbols.
    */
    public static Set<Symbol> getUseSymbol(Traversable t) {
        Set<Symbol> ret = new HashSet<Symbol>();
        for (Expression e : getUseSet(t)) {
            Symbol symbol = SymbolTools.getSymbolOf(e);
            if (symbol != null) {
                ret.add(symbol);
            }
        }
        return ret;
    }

    /**
    * Returns a set of used symbols from the traversable object.
    *
    * @param t the traversable object.
    * @return the set of used symbols.
    */
    private static Set<Symbol> getFlatUseSymbol(Traversable t) {
        Set<Symbol> ret = new HashSet<Symbol> ();
        for (Expression e : getFlatUseSet(t)) {
            Symbol symbol = SymbolTools.getSymbolOf(e);
            if (symbol != null) {
                ret.add(symbol);
            }
        }
        return ret;
    }

    public static List<Expression> getUseList(Traversable t) {
        List<Expression> ret = new ArrayList<Expression>();
        DFIterator<Expression> iter =
                new DFIterator<Expression>(t, Expression.class);
        // Handle these expressions specially.
        iter.pruneOn(AccessExpression.class);
        iter.pruneOn(ArrayAccess.class);
        iter.pruneOn(AssignmentExpression.class);
        while (iter.hasNext()) {
            Expression o = iter.next();
            if (o instanceof AccessExpression) {
                AccessExpression ae = (AccessExpression)o;
                DFIterator<ArrayAccess> ae_iter =
                        new DFIterator<ArrayAccess>(ae, ArrayAccess.class);
                ae_iter.pruneOn(ArrayAccess.class);
                // Catches array subscripts in the access expression.
                while (ae_iter.hasNext()) {
                    ArrayAccess aa = ae_iter.next();
                    List<Expression> aa_use = getUseList(aa);
                    aa_use.remove(aa);
                    ret.addAll(aa_use);
                }
                ret.add(ae);
            } else if (o instanceof ArrayAccess) {
                ArrayAccess aa = (ArrayAccess)o;
                for (int i = 0; i < aa.getNumIndices(); ++i) {
                    ret.addAll(getUseList(aa.getIndex(i)));
                }
                ret.add(aa);
            } else if (o instanceof AssignmentExpression) {
                AssignmentExpression ae = (AssignmentExpression)o;
                ret.addAll(getUseList(ae.getRHS()));
                List<Expression> lhs_use = getUseList(ae.getLHS());
                // Other cases should include the lhs in the used set. (+=,...)
                if (ae.getOperator() == AssignmentOperator.NORMAL) {
                    lhs_use.remove(ae.getLHS());
                }
                ret.addAll(lhs_use);
            } else if (o instanceof Identifier) {
                Identifier id = (Identifier)o;
                if (id.getSymbol() instanceof Procedure ||
                    id.getSymbol() instanceof ProcedureDeclarator) {
                    ;
                } else {
                    ret.add(id);
                }
            } else if (o instanceof UnaryExpression) {
                UnaryExpression ue = (UnaryExpression)o;
                UnaryOperator uop = ue.getOperator();
                if (uop == UnaryOperator.POST_DECREMENT ||
                    uop == UnaryOperator.POST_INCREMENT ||
                    uop == UnaryOperator.PRE_DECREMENT ||
                    uop == UnaryOperator.PRE_INCREMENT) {
                    ret.addAll(getUseList(ue.getExpression()));
                } else {
                    ret.add((UnaryExpression)o);
                }
            }
        }
        return ret;
    }

}
