/* 
  OpenMP spec 3.0 (the last section in page 99)
  
  The restrictions to the reduction clause are as follows 
   * A list item that appears in a reduction clause of a worksharing construct 
     must be shared in the parallel regions to which any of the worksharing 
     regions arising from the worksharing construct bind.
   * A list item that appears in a reduction clause of the innermost enclosing
     worksharing or parallel construct may not be accessed in an explicit task.
   * Any number of reduction clauses can be specified on the directive, but a 
     list item can appear only once in the reduction clause(s) for that
     directive.
     
  C/C++ specific restrictions
   * A type of a list item that appears in a reduction clause must be valid 
     for the reduction operator.
   * Aggregate types (including arrays), pointer types and reference types 
     may not appear in a reduction clause. 
   * A list item that appears in a reduction clause must not be
     const-qualified.
   * The operator specified in a reduction clause cannot be overloaded with 
     respect to C/C++ the list items that appear in that clause
*/
/**
* Reduction pass performs reduction recognition for each ForLoop.
* It generates cetus annotation in the form of "#pragma cetus reduction(...)"
* Currently, it supports scalar (sum += ...), ArrayAccess (A[i] += ...),  and 
* AccessExpression (A->x += ...) for reduction variable.
* If another pass wants to access reduction information for a given statement,
* stmt, Tools.getAnnotation(stmt, "cetus", "reduction") will return an object 
* that contains a reduction map.
* A reduction map, rmap, is a HashMap and has the following form;
* <code>
*    Map<String, Set<Expression>> rmap;
* </code>
* (where String represents a reduction operator and Set<Expression> is a 
* set of reduction variables)
*/

package cetus.analysis;

import cetus.exec.Driver;
import cetus.hir.*;
import java.util.*;

/**
* Performs reduction variable analysis to detect and annotate statements like
* x = x + i in loops. An Annotation is added right before loops that contain
* reduction variables. 
*/
public class Reduction extends AnalysisPass {

    static int debug_tab = 0;

    private static final int SCALAR_REDUCTION = 1;

    private int debug_level;

    private AliasAnalysis alias;

    private int option;

    private static final String pass_name = "[Reduction]";

    public Reduction(Program program) {
        super(program);
        debug_level = PrintTools.getVerbosity();
        try {
            option = Integer.parseInt(Driver.getOptionValue("reduction"));
        } catch(NumberFormatException e) {
            option = SCALAR_REDUCTION;
        }
    }

    public String getPassName() {
        return pass_name;
    }

    public void start() {
        alias = new AliasAnalysis(program);
        alias.start();
/*
        DDTDriver ddtest = new DDTDriver(program);
        ddtest.start();
*/
        DFIterator<ForLoop> iter =
                new DFIterator<ForLoop>(program, ForLoop.class);
        while (iter.hasNext()) {
            ForLoop loop = iter.next();
            // find reduction variables in a loop
            Map<String, Set<Expression>> reduce_map = analyzeStatement(loop);
            // Insert reduction Annotation to the current loop
            if (!reduce_map.isEmpty()) {
                CetusAnnotation note =
                        new CetusAnnotation("reduction", reduce_map);
                loop.annotateBefore(note);
            }
        }
    }

    public void displayMap(Map<Symbol, Set<Integer>> imap, String name) {
        if (debug_level > 2) {
            int key_cnt = 0;
            for (Symbol sym : imap.keySet()) {
                if (sym == null) {
                    continue;
                }
                System.out.print(name + ++key_cnt + " : " +
                                 sym.getSymbolName() + " = {");
                int val_cnt = 0;
                for (Integer hashcode : imap.get(sym)) {
                    if (val_cnt++ == 0) {
                        System.out.print(hashcode.toString());
                    } else {
                        System.out.print(", " + hashcode.toString());
                    }
                }
                System.out.println("}");
            }
        }
    }

    // Reduction recognition on statements including ForLoop, CompoundStatement,
    // and a Statement
    public Map<String, Set<Expression>> analyzeStatement(Statement istmt) {
        debug_tab++;
        if (debug_level > 1) {
            System.out.println(
                    "------------ analyzeStatement strt ------------\n");
        }
        // rmap: a map of
        // <a reduction operator, a set of reduction candidate variable>
        Map<String, Set<Expression>> rmap =
                new HashMap<String, Set<Expression>>();
        // cmap: a map that contains candidate reduction variables
        Map<Symbol, Set<Integer>> cmap = new HashMap<Symbol, Set<Integer>>();
        // RefMap: Referenced variable set
        Map<Symbol, Set<Integer>> UseMap = DataFlowTools.getUseSymbolMap(istmt);
        Map<Symbol, Set<Integer>> DefMap = DataFlowTools.getDefSymbolMap(istmt);
        displayMap(UseMap, "UseMap");
        displayMap(DefMap, "DefMap");
        Map<Symbol, Set<Integer>> RefMap = new HashMap<Symbol, Set<Integer>>();
        RefMap.putAll(UseMap);
        DataFlowTools.mergeSymbolMaps(RefMap, DefMap);
        Set<Symbol> side_effect_set = new HashSet<Symbol>();
        int expr_cnt = 0;
        DFIterator<Expression> iter =
                new DFIterator<Expression>(istmt, Expression.class);
        iter.pruneOn(AlignofExpression.class);
        iter.pruneOn(ArrayAccess.class);
        iter.pruneOn(IDExpression.class);
        iter.pruneOn(InfExpression.class);
        iter.pruneOn(Literal.class);
        iter.pruneOn(NewExpression.class);
        iter.pruneOn(SizeofExpression.class);
        while (iter.hasNext()) {
            Expression expr = iter.next();
            PrintTools.printlnStatus(9, pass_name, "[expr]", ++expr_cnt, ":",
                    expr, "(", expr.getClass().getName(), ")");
            if (expr instanceof AssignmentExpression) {
                AssignmentExpression assign_expr = (AssignmentExpression)expr;
                findReduction(assign_expr, rmap, cmap);
            } else if (expr instanceof UnaryExpression) {
                UnaryExpression unary_expr = (UnaryExpression)expr;
                findReduction(unary_expr, rmap, cmap);
            } else if (expr instanceof FunctionCall) {
                Set<Symbol> func_side_effect =
                        SymbolTools.getSideEffectSymbols((FunctionCall)expr);
                displaySet("side_effect_set(" + expr.toString() + ")",
                           func_side_effect);
                side_effect_set.addAll(func_side_effect);
            }
        }
        // if the lhse of the reduction candidate statement is not in the
        // RefMap, lhse is a reduction variable
        displayMap(RefMap, "RefMap");
        displayMap(cmap, "cmap");
        // Remove expressions used as reduction variables from RefMap
        // Foreach reduction operator ("+" and "*")
        for (String op : rmap.keySet()) {
            for (Expression candidate : rmap.get(op)) {
                Symbol candidate_symbol = SymbolTools.getSymbolOf(candidate);
                Set<Integer> reduceSet = cmap.get(candidate_symbol);
                Set<Integer> referenceSet = RefMap.get(candidate_symbol);
                if (referenceSet == null) {
                    continue;
                }
                referenceSet.removeAll(reduceSet);
            }
        }
        // final reduction map that maps a reduction operator to a set of
        // reduction variables
        Map<String, Set<Expression>> fmap =
                new HashMap<String, Set<Expression>>();
        // Foreach reduction operator ("+" and "*")
        for (String op : rmap.keySet()) {
            for (Expression candidate : rmap.get(op)) {
                boolean remove_flag = false;
                PrintTools.printlnStatus(2, pass_name, "candidate:", candidate);
                Symbol candidate_symbol = SymbolTools.getSymbolOf(candidate);
                if (RefMap.get(candidate_symbol) == null) {
                    continue;
                }
                if (!RefMap.get(candidate_symbol).isEmpty()) {
                    PrintTools.printlnStatus(2, pass_name, candidate,
                            "is referenced in the non-reduction statement!");
                    remove_flag = true;
                }
                if (alias != null) {
                    DFIterator<Statement> stmt_iter =
                            new DFIterator<Statement>(istmt, Statement.class);
                    while (stmt_iter.hasNext()) {
                        Statement o = stmt_iter.next();
                        if (o instanceof ExpressionStatement ||
                            o instanceof DeclarationStatement ||
                            o instanceof ReturnStatement) {
                            if (alias.isAliased(
                                        o, candidate_symbol, RefMap.keySet())) {
                                PrintTools.printlnStatus(2, pass_name,
                                        candidate, "is Aliased!");
                                remove_flag = true;
                                break;
                            }
                        }
                    }
                }
                if (side_effect_set.contains(candidate_symbol)) {
                    PrintTools.printlnStatus(2, pass_name,
                            candidate, "has side-effect!");
                    remove_flag = true;
                }
                if (candidate instanceof ArrayAccess &&
                    istmt instanceof ForLoop) {
                    // check if a candidate has a self carried loop dependence
                    // from the DD graph
                    // for (i=0; i<N; i++) { A[i] += expr; } :
                    //              A[i] is not a reduction
/*
                    if (!program.ddgraph.checkSelfLoopCarriedDependence(
                            candidate, loop)) {
                        PrintTools.println(
                                "No self-carried output dependence in " +
                                candidate, 2);
                        PrintTools.println("loop: " + loop, 2);
                        reduction_set.remove(candidate);     
                    }
*/
                    if (simple_self_dependency_check(
                            (ArrayAccess)candidate, (ForLoop)istmt)) {
                        PrintTools.printlnStatus(2, pass_name,
                                "No self-carried-output dependence in",
                                candidate);
                        remove_flag = true;
                    }
                    if (option == SCALAR_REDUCTION) {
                        remove_flag = true;
                    }
                }
                if (remove_flag == false) {
                    if (fmap.containsKey(op)) {
                        fmap.get(op).add(candidate);
                    } else {
                        Set<Expression> new_set = new HashSet<Expression>();
                        new_set.add(candidate);
                        fmap.put(op, new_set);
                    }
                }
            }
        }
        if (debug_level > 1) {
            print_reduction(fmap);
            System.out.println(
                    "------------ analyzeStatement done ------------\n");
        }
        debug_tab--;
        return fmap;
    }

    /**
    * Reduction recognition on OpenMP critical sections including ForLoop,
    * CompoundStatement, and a Statement. This method is the same as
    * analyzeStatement() except that this method skips self-carried-output
    * dependency checking part; analyzeStatement() works on outer-most loops,
    * but this method works on inner-loops (loops contained in a critical
    * section of a parallel region can be considered as inner-loops with
    * respect to the enclosing parallel region).
    * @param istmt critical sections 
    * @return mapping of (reduction operator, reduction variables)
    */
    public Map<String, Set<Expression>> analyzeStatement2(Statement istmt) {
        debug_tab++;
        if (debug_level > 1) {
            System.out.println(
                    "------------ analyzeStatement strt ------------\n");
        }
        // rmap: a map of
        // <a reduction operator, a set of reduction candidate variable>
        Map<String, Set<Expression>> rmap =
                new HashMap<String, Set<Expression>>();
        // cmap: a map that contains candidate reduction variables
        Map<Symbol, Set<Integer>> cmap = new HashMap<Symbol, Set<Integer>>();
        // RefMap: Referenced variable set
        Map<Symbol, Set<Integer>> UseMap = DataFlowTools.getUseSymbolMap(istmt);
        Map<Symbol, Set<Integer>> DefMap = DataFlowTools.getDefSymbolMap(istmt);
        displayMap(UseMap, "UseMap");
        displayMap(DefMap, "DefMap");
        Map<Symbol, Set<Integer>> RefMap = new HashMap<Symbol, Set<Integer>>();
        RefMap.putAll(UseMap);
        DataFlowTools.mergeSymbolMaps(RefMap, DefMap);
        Set<Symbol> side_effect_set = new HashSet<Symbol>();
        int expr_cnt = 0;
        DFIterator<Expression> iter =
                new DFIterator<Expression>(istmt, Expression.class);
        iter.pruneOn(AlignofExpression.class);
        iter.pruneOn(ArrayAccess.class);
        iter.pruneOn(IDExpression.class);
        iter.pruneOn(InfExpression.class);
        iter.pruneOn(Literal.class);
        iter.pruneOn(NewExpression.class);
        iter.pruneOn(SizeofExpression.class);
        while (iter.hasNext()) {
            Expression expr = iter.next();
            PrintTools.printlnStatus(9, pass_name, "[expr]", ++expr_cnt,
                    ":", expr, "(", expr.getClass().getName(), ")");
            if (expr instanceof AssignmentExpression) {
                AssignmentExpression assign_expr = (AssignmentExpression)expr;
                findReduction(assign_expr, rmap, cmap);
            } else if (expr instanceof UnaryExpression) {
                UnaryExpression unary_expr = (UnaryExpression)expr;
                findReduction(unary_expr, rmap, cmap);
            } else if (expr instanceof FunctionCall) {
                Set<Symbol> func_side_effect =
                        SymbolTools.getSideEffectSymbols((FunctionCall)expr);
                displaySet("side_effect_set(" + expr.toString() + ")",
                           func_side_effect);
                side_effect_set.addAll(func_side_effect);
            }
        }
        // if the lhse of the reduction candidate statement is not in the
        // RefMap, lhse is a reduction variable
        displayMap(RefMap, "RefMap");
        displayMap(cmap, "cmap");
        // Remove expressions used as reduction variables from RefMap
        // Foreach reduction operator ("+" and "*")
        for (String op : rmap.keySet()) {
            for (Expression candidate : rmap.get(op)) {
                Symbol candidate_symbol = SymbolTools.getSymbolOf(candidate);
                Set<Integer> reduceSet = cmap.get(candidate_symbol);
                Set<Integer> referenceSet = RefMap.get(candidate_symbol);
                if (referenceSet == null) {
                    continue;
                }
                referenceSet.removeAll(reduceSet);
            }
        }
        // final reduction map that maps a reduction operator to a set of
        // reduction variables
        Map<String, Set<Expression>> fmap =
                new HashMap<String, Set<Expression>>();
        // Foreach reduction operator ("+" and "*")
        for (String op:rmap.keySet()) {
            for (Expression candidate : rmap.get(op)) {
                boolean remove_flag = false;
                PrintTools.printlnStatus(2, pass_name, "candidate:", candidate);
                Symbol candidate_symbol = SymbolTools.getSymbolOf(candidate);
                if (RefMap.get(candidate_symbol) == null) {
                    continue;
                }
                if (!RefMap.get(candidate_symbol).isEmpty()) {
                    PrintTools.printlnStatus(2, pass_name, candidate,
                            "is referenced in the non-reduction statement!");
                    remove_flag = true;
                }
                if (alias != null) {
                    DFIterator<Statement> stmt_iter =
                            new DFIterator<Statement>(istmt, Statement.class);
                    while (stmt_iter.hasNext()) {
                        Statement o = stmt_iter.next();
                        if (o instanceof ExpressionStatement ||
                            o instanceof DeclarationStatement ||
                            o instanceof ReturnStatement) {
                            if (alias.isAliased(
                                    o, candidate_symbol, RefMap.keySet())) {
                                PrintTools.printlnStatus(2, pass_name,
                                        candidate, "is Aliased!");
                                remove_flag = true;
                                break;
                            }
                        }
                    }
                }
                if (side_effect_set.contains(candidate_symbol)) {
                    PrintTools.printlnStatus(2, pass_name,
                            candidate, "has side-effect!");
                    remove_flag = true;
                }
                if (remove_flag == false) {
                    if (fmap.containsKey(op)) {
                        fmap.get(op).add(candidate);
                    } else {
                        Set<Expression> new_set = new HashSet<Expression>();
                        new_set.add(candidate);
                        fmap.put(op, new_set);
                    }
                }
            }
        }
        if (debug_level > 1) {
            print_reduction(fmap);
            System.out.println(
                    "------------ analyzeStatement done ------------\n");
        }
        debug_tab--;
        return fmap;
    }

    private void add_to_cmap(
            Map<Symbol, Set<Integer>> cmap, Symbol reduce_sym, Integer hcode) {
        if (cmap.containsKey(reduce_sym)) {
            Set<Integer> HashCodeSet = cmap.get(reduce_sym);
            HashCodeSet.add(hcode);
        } else {
            Set<Integer> HashCodeSet = new HashSet<Integer>();
            HashCodeSet.add(hcode);
            cmap.put(reduce_sym, HashCodeSet);
        }
    }

    private void add_to_rmap(Map<String, Set<Expression>> rmap,
                             String reduce_op, Expression reduce_expr) {
        Set<Expression> reduce_set;
        if (rmap.keySet().contains(reduce_op)) {
            reduce_set = rmap.get(reduce_op);
            rmap.remove(reduce_op);
        } else {
            reduce_set = new HashSet<Expression>();
        }
        if (!reduce_set.contains(reduce_expr)) {
            reduce_set.add(reduce_expr);
        }
        rmap.put(reduce_op, reduce_set);
    }

    private void findReduction(UnaryExpression expr,
                               Map<String, Set<Expression>> rmap,
                               Map<Symbol, Set<Integer>> cmap) {
        boolean isReduction = false;
        UnaryOperator unary_op = expr.getOperator();
        Expression lhse = expr.getExpression();
        String reduction_op = null;
        if (lhse instanceof IDExpression || lhse instanceof ArrayAccess ||
            lhse instanceof AccessExpression) {
            if (unary_op == UnaryOperator.PRE_INCREMENT ||
                unary_op == UnaryOperator.POST_INCREMENT) {
                reduction_op = new String("+");
                isReduction = true;
            } else if (unary_op == UnaryOperator.PRE_DECREMENT ||
                       unary_op == UnaryOperator.POST_DECREMENT) {
                reduction_op = new String("+");
                isReduction = true;
            }
        }
        if (isReduction) {
            add_to_rmap(rmap, reduction_op, lhse);
            add_to_cmap(cmap, SymbolTools.getSymbolOf(lhse),
                        System.identityHashCode(lhse));
            PrintTools.printlnStatus(2, pass_name,
                    "candidate = (", reduction_op, ":", lhse, ")");
        }
    }

    private void findReduction(AssignmentExpression expr,
                               Map<String, Set<Expression>> rmap,
                               Map<Symbol, Set<Integer>> cmap) {
        boolean isReduction = false;
        AssignmentOperator assign_op = expr.getOperator();
        Expression lhse = expr.getLHS();
        Expression rhse = expr.getRHS();
        Expression lhse_removed_rhse = null;
        String reduction_op = null;
        if (lhse instanceof IDExpression || lhse instanceof ArrayAccess ||
            lhse instanceof AccessExpression) {
            if (assign_op == AssignmentOperator.NORMAL) {
                // at this point either "lhse = expr;" or "lhse = lhse + expr;"
                // is possible
                Expression simplified_rhse = Symbolic.simplify(rhse);
                Expression lhse_in_rhse =
                        IRTools.findExpression(simplified_rhse, lhse);
                // if it is null, then it is not a reduction statement
                if (lhse_in_rhse == null) {
                    return;
                }
                Expression parent_expr = (Expression)lhse_in_rhse.getParent();
                if (parent_expr instanceof BinaryExpression) {
                    reduction_op =
                      ((BinaryExpression)parent_expr).getOperator().toString();
                    if (reduction_op.equals("+")) {
                        lhse_removed_rhse = Symbolic.subtract(rhse, lhse);
                    } else if (reduction_op.equals("*")) {
                        lhse_removed_rhse = Symbolic.divide(rhse, lhse);
                    } else {
                        // operators, such as {&, |, ^, &&, ||}
                        // are not supported
                        return;
                    }
                } else {
                    return;
                }
            } else if ((assign_op == AssignmentOperator.ADD) ||
                       (assign_op == AssignmentOperator.SUBTRACT)) {
                // case: lhse += expr; or lhse -= expr; 
                lhse_removed_rhse = Symbolic.simplify(rhse);
                if (lhse_removed_rhse == null) {
                    System.out.println("[+= or -=] rhse_removed_rhse is null");
                }
                reduction_op = "+";
            } else if (assign_op == AssignmentOperator.MULTIPLY) {
                // case: lhse *= expr;
                lhse_removed_rhse = Symbolic.simplify(rhse);
                if (lhse_removed_rhse == null) {
                    System.out.println("[*=] rhse_removed_rhse is null");
                }
                reduction_op = "*";
            } else {
                return;
            }
            if (debug_level > 1) {
                if (lhse_removed_rhse == null) {
                    System.out.println("[ERROR] rhse_removed_rhse is null");
                }
                System.out.println("lhse_removed_rhse=" + lhse_removed_rhse);
            }
            if (lhse instanceof Identifier) {
                Identifier id = (Identifier)lhse;
                if (!IRTools.containsSymbol(lhse_removed_rhse, id.getSymbol())){
                    isReduction = true;
                }
            } else if (lhse instanceof ArrayAccess) {
                Expression base_array_name = ((ArrayAccess)lhse).getArrayName();
                if (base_array_name instanceof Identifier) {
                    Identifier id = (Identifier)base_array_name;
                    if (!IRTools.containsSymbol(
                                lhse_removed_rhse, id.getSymbol())) {
                        isReduction = true;
                    }
                }
            } else if (lhse instanceof AccessExpression) {
                Symbol lhs_symbol = SymbolTools.getSymbolOf(lhse);
                if (!IRTools.containsSymbol(lhse_removed_rhse, lhs_symbol)) {
                    isReduction = true;
                }
            }
        }
        if (isReduction) {
            add_to_rmap(rmap, reduction_op, lhse);
            for (Expression e : IRTools.findExpressions(expr, lhse)) {
                add_to_cmap(cmap, SymbolTools.getSymbolOf(lhse),
                        System.identityHashCode(e));
            }
            PrintTools.printlnStatus(2, pass_name,
                    "candidate = (", reduction_op, ":", lhse, ")");
        }
    }

    public void print_reduction(Map<String, Set<Expression>> map) {
        if (debug_level < 1) {
            return;
        }
        if (!map.isEmpty()) {
            int op_cnt = 0;
            StringBuilder sb = new StringBuilder();
            sb.append("reduction = {");
            for (String op : map.keySet()) {
                int cnt = 0;
                if (op_cnt++ > 0) {
                    sb.append(", ");
                }
                sb.append("(").append(op).append(":");
                for (Expression expr : map.get(op)) {
                    if (cnt++ > 0) {
                        sb.append(", ");
                    }
                    sb.append(expr);
                }
                sb.append(")");
            }
            sb.append("}").append(PrintTools.line_sep);
            PrintTools.printlnStatus(1, pass_name, sb);
        } else {
            PrintTools.printlnStatus(1, pass_name, "reduction = {}");
        }
    }

    /**
    * returns true if an array access index are all IntegerLiteral, eg, A[2][3].
    */
    private boolean
            is_an_array_element_with_constant_index(ArrayAccess expr) {
        for (int i = 0; i < expr.getNumIndices(); i++) {
            if (!(expr.getIndex(i) instanceof IntegerLiteral)) {
                return false;
            }
        }
        return true;
    }

    private boolean
            simple_self_dependency_check(ArrayAccess aae, Loop loop) {
        Expression loop_index_expr = LoopTools.getIndexVariable(loop);
        Symbol loop_index_symbol = SymbolTools.getSymbolOf(loop_index_expr);
        if (loop_index_symbol != null) {
            for (int i = 0; i < aae.getNumIndices(); i++) {
                Expression array_index_expr = aae.getIndex(i);
                Symbol array_index_symbol =
                        SymbolTools.getSymbolOf(array_index_expr);
                if (array_index_symbol != null) {
                    if (loop_index_symbol == array_index_symbol) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    public void displaySet(String name, Set iset) {
        int cnt = 0;
        if (iset == null) {
            return;
        }
        if (debug_level <= 1) {
            return;
        }
        System.out.print(name + ":");
        for (Object o : iset) {
            if ((cnt++) == 0) {
                System.out.print("{");
            } else {
                System.out.print(", ");
            }
            if (o instanceof Expression) {
                System.out.print(o.toString());
            } else if (o instanceof Symbol) {
                System.out.print(((Symbol)o).getSymbolName());
            } else {
                if (o == null) {
                    System.out.println("null");
                } else {
                    System.out.println("obj: " + o.getClass().getName());
                }
            }
        }
        System.out.println("}");
    }
}
