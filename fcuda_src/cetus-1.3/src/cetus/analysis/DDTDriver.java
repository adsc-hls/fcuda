package cetus.analysis;

import cetus.hir.*;
import java.util.*;

/*
Algorithm:
---- Traverse IR ----
Traverse across all loops in the program
Collect outermost loops that are eligible for dependence testing
Eligibility is determined by LoopTools and is currently as defined below

Eligibility:
  - Canonical loops (Fortran type DO loop format) - See LoopTools.java for
    details
  - Cannot contain function calls
  - Currently allows function calls that can be parallelized (hard-coded or
    determined from simple side effect analysis - see LoopTools.java)
  - Loops with control flow modifiers are tested for dependence but handled
    correctly during parallelization
  - Loops with symbolic increments whose values cannot be determined are 
    considered ineligible

----- Collect Info -----
if(eligible)
  collect loop information
    - loop bounds, increments, index variable
    - array accesses in entire nest

----- Run test ----
Test for data dependence between accesses of the same array within the entire
loop nest, obtain direction vector set and build a Data Dependence Graph (DDG)
  - For aliased variables, assume dependences conservatively
*/

/**
* Performs array data-dependence analysis on eligible loop nests in the program
*/
public class DDTDriver extends AnalysisPass {
    /* Member variables */

    private static final String pass_name = "[DDT]";

    /* Alias analysis related to Dependence testing */
    private AliasAnalysis alias_analysis;

    /* verbosity */
    private int verbosity = PrintTools.getVerbosity();

    /**
    * Constructor
    */
    public DDTDriver(Program program) {
        super(program);
    }

    /** 
    * Performs whole program data-dependence analysis 
    */
    public void start() {
        LoopTools.addLoopName(program);
        program.createNewDDGraph();
        DDGraph ddg = program.getDDGraph();
        // Run Alias Analysis as currently it has been implemented as whole
        // program analysis
        alias_analysis = new AliasAnalysis(program);
        AnalysisPass.run(alias_analysis);
        // Obtain a list of loops that enclose eligible nests for dependence
        // testing
        List<Loop> eligible_loops =
                LoopTools.extractOutermostDependenceTestEligibleLoops(program);
        PrintTools.printlnStatus(1, pass_name,
                "Number of eligible outermost loops =", eligible_loops.size());
        for (Loop loop : eligible_loops) {
            DDGraph dependence_graph = analyzeLoopsForDependence(loop);
            ddg.addAllArcs(dependence_graph.getAllArcs());
        }
        // TODO: replace with generic metric
        //PrintTools.println("Size of data dependence graph = " +
        //                   (ddg.getAllArcs()).size(), 1);
    }

    public String getPassName() {
        return pass_name;
    }

    /**
    * Analyze this loop and all inner loops to build a dependence graph for the
    * loop nest with this loop as the outermost loop.
    */
    public DDGraph analyzeLoopsForDependence(Loop loop) {
        // Dependence Graph for nest defined by this loop
        DDGraph loopDDGraph = new DDGraph();
        // Storage space for loop information
        HashMap<Loop, LoopInfo> loopInfoMap = new HashMap<Loop, LoopInfo>();
        // Map from array name (Expression) to array access information
        // (DDArrayAccessInfo)
        HashMap<Symbol, ArrayList<DDArrayAccessInfo>> loopArrayAccessMap =
            new HashMap<Symbol, ArrayList<DDArrayAccessInfo>>();
        // Collect loop information and array access information
        LinkedList<Loop> nest = LoopTools.calculateInnerLoopNest(loop);
        for (Loop l : nest) {
            collectLoopInfo(l, loopInfoMap, loopArrayAccessMap);
        }
        // Run Data-Dependence Tests for entire nest and return the DDG if
        // everything went OK
        loopDDGraph = runDDTest(loop, loopArrayAccessMap, loopInfoMap);
        return loopDDGraph;
    }

    /**
    * Performs data-dependence analysis on the nest defined by loop and returns
    * the dependence graph
    * summarize = Direction Vectors summarized or not
    * @return DDGraph - dependence graph for nest
    *            null if loop nest is not eligible for dependence analysis
    * @deprecated
    * This routine must no longer be used to get access to data dependence
    * information. Instead, obtain a copy of the dependence graph from Program
    * object and use DDGraph API for details related to dependence information
    */
    @Deprecated
    public DDGraph getDDGraph(boolean summarize, Loop loop) {
        DDGraph loopDDGraph = analyzeLoopsForDependence(loop);
        if (loopDDGraph != null) {
            if (summarize) {
                loopDDGraph.summarizeGraph();
                return loopDDGraph;
            } else {
                return loopDDGraph;
            }
        }
        // Loop nest was not eligible for dependence analysis
        else {
            return null;
        }
    }

    /**
    * Analyze every loop to gather and store loop information
    * such as upper and lower bounds, increment, index variable.
    * Also build framework required for calling DD tests
    * 
    * @return boolean
    */
    private void collectLoopInfo(
            Loop currentLoop,
            HashMap<Loop, LoopInfo> loopInfoMap,
            HashMap<Symbol, ArrayList<DDArrayAccessInfo>> loopArrayAccessMap) {
        // Gather loop information and store it in a map with the loop as the
        // key
        LoopInfo loop_info = new LoopInfo(currentLoop);
        // Use range analysis information to remove symbolic values from loop
        // information
        RangeDomain loop_range = RangeAnalysis.query((Statement)currentLoop);
        // Lower bound for loop is not constant, use range information
        if (!(LoopTools.isLowerBoundConstant(currentLoop))) {
            Expression new_lb = LoopTools.replaceSymbolicLowerBound(
                    currentLoop, loop_range);
            // Assign new lower bound to loop in Loop Info
            loop_info.setLoopLB(new_lb);
        }
        // Upper bound for loop is not constant, use range information
        if (!(LoopTools.isUpperBoundConstant(currentLoop))) {
            Expression new_ub = LoopTools.replaceSymbolicUpperBound(
                    currentLoop, loop_range);
            // Assign new upper bound to loop in Loop Info
            loop_info.setLoopUB(new_ub);
        }
        // Increment for loop is not constant, use range information
        // Range information will return constant integer increment value as
        // the loop has already been considered eligible for dependence testing
        if (!(LoopTools.isIncrementConstant(currentLoop))) {
            Expression curr_inc = loop_info.getLoopIncrement();
            Set<Symbol> loop_stmt_symbols = loop_range.getSymbols();
            Expression new_inc = loop_range.substituteForward(
                    curr_inc, loop_stmt_symbols);
            loop_info.setLoopIncrement(new_inc);
        }
        // Finally, attach updated loop info to the map
        loopInfoMap.put(currentLoop, loop_info);
        // get write and read array accesses only for this loop body
        addWriteAccesses(currentLoop,loopArrayAccessMap,currentLoop.getBody());
        addReadAccesses(currentLoop,loopArrayAccessMap,currentLoop.getBody());
    }

    private void addWriteAccesses(
            Loop loop,
            HashMap<Symbol, ArrayList<DDArrayAccessInfo>> loopArrayAccessMap,
            Traversable root) {
        if (root instanceof Expression) {
            DFIterator<Expression> iter =
                    new DFIterator<Expression>(root, Expression.class);
            iter.pruneOn(AssignmentExpression.class);
            iter.pruneOn(UnaryExpression.class);
            iter.pruneOn(FunctionCall.class);
            while (iter.hasNext()) {
                Expression o = iter.next();
                if (o instanceof AssignmentExpression) {
                    AssignmentExpression expr = (AssignmentExpression)o;
                    //
                    // Only the left-hand side of an AssignmentExpression
                    // is a definition.  There may be other nested
                    // definitions but, since iter is not set to prune
                    // on AssignmentExpressions, they will be found during
                    // the rest of the traversal.
                    //
                    addWriteAccesses(loop, loopArrayAccessMap, expr.getLHS());
                    /*
                    if (expr.getLHS() instanceof ArrayAccess) {
                        Statement stmt = (expr.getLHS()).getStatement();
                        DDArrayAccessInfo arrayInfo = new DDArrayAccessInfo(
                                (ArrayAccess)expr.getLHS(),
                                DDArrayAccessInfo.write_type, loop, stmt);
                        addArrayAccess(arrayInfo, loopArrayAccessMap);
                    }
                    */
                } else if (o instanceof UnaryExpression) {
                    UnaryExpression expr = (UnaryExpression) o;
                    UnaryOperator op = expr.getOperator();
                    //
                    // there are only a few UnaryOperators that create
                    // definitions
                    //
                    if (UnaryOperator.hasSideEffects(op) &&
                        expr.getExpression() instanceof ArrayAccess) {
                        Statement stmt = (expr.getExpression()).getStatement();
                        DDArrayAccessInfo arrayInfo = new DDArrayAccessInfo(
                                (ArrayAccess)expr.getExpression(),
                                DDArrayAccessInfo.write_type, loop, stmt);
                        addArrayAccess(arrayInfo, loopArrayAccessMap);
                    }
                } else if (o instanceof FunctionCall) {
                    List<Expression> arguments =
                            ((FunctionCall)o).getArguments();
                    for (Expression e : arguments) {
                        if ((e instanceof UnaryExpression) &&
                            ((UnaryExpression)e).getOperator() ==
                                UnaryOperator.ADDRESS_OF &&
                            ((UnaryExpression)e).getExpression()
                                instanceof ArrayAccess) {
                            addWriteAccesses(loop, loopArrayAccessMap,
                                    ((UnaryExpression)e).getExpression());
                        }
                    }
                } else if (o instanceof ArrayAccess) {
                    ArrayAccess acc = (ArrayAccess)o;
                    Statement stmt = acc.getStatement();
                    DDArrayAccessInfo arrayInfo = new DDArrayAccessInfo(
                            acc, DDArrayAccessInfo.write_type, loop, stmt);
                    addArrayAccess(arrayInfo, loopArrayAccessMap);
                }
            }
        } else if (root instanceof IfStatement) {
            IfStatement if_stmt = (IfStatement)root;
            addWriteAccesses(loop, loopArrayAccessMap,
                    if_stmt.getThenStatement());
            if (if_stmt.getElseStatement() != null) {
                addWriteAccesses(loop, loopArrayAccessMap,
                    if_stmt.getElseStatement());
            }
        } else if (root instanceof Loop) {
        } else if (root instanceof DeclarationStatement) {
            // need to skip because comments are DeclarationStatement
        } else {
            List<Traversable> children = root.getChildren();
            for (int i = 0; i < children.size(); i++) {
                addWriteAccesses(loop, loopArrayAccessMap, children.get(i));
            }
        }
    }

    private void addReadAccesses(
            Loop loop,
            HashMap<Symbol, ArrayList<DDArrayAccessInfo>> loopArrayAccessMap,
            Traversable root) {
        DFIterator<Expression> iter =
                new DFIterator<Expression>(root, Expression.class);
        iter.pruneOn(AccessExpression.class);
        iter.pruneOn(AssignmentExpression.class);
        iter.pruneOn(Loop.class);
        while (iter.hasNext()) {
            Expression o = iter.next();
            if (o instanceof AssignmentExpression) {
                AssignmentExpression expr = (AssignmentExpression) o;
                // Recurse on the right-hand side because it is being read.
                addReadAccesses(loop, loopArrayAccessMap, expr.getRHS());
                //
                //  The left-hand side also may have uses, but unless the
                //  assignment is an update like +=, -=, etc. the top-most
                //  left-hand side expression is a definition and not a use.
                //
                if (expr.getOperator() != AssignmentOperator.NORMAL) {
                    addReadAccesses(loop, loopArrayAccessMap, expr.getLHS());
                }
            } else if (o instanceof AccessExpression) {
                AccessExpression expr = (AccessExpression)o;
                //
                //  The left-hand side of an access expression
                //  is read in the case of p->field.  For accesses
                //  like p.field, we still consider it to be a use
                //  of p because it could be a use in C++ or Java
                //  (because p could be a reference) and it doesn't
                //  matter for analysis of C (because it will never
                //  be written.
                //
                addReadAccesses(loop, loopArrayAccessMap, expr.getLHS());
                /*
                Statement stmt = ((Expression)o).getStatement();
                DDArrayAccessInfo arrayInfo = new DDArrayAccessInfo(
                    (ArrayAccess)(o), DDArrayAccessInfo.read_type, loop, stmt);
                addArrayAccess(arrayInfo, loopArrayAccessMap);
                */
            } else if (o instanceof Loop) {
            } else if (o instanceof ArrayAccess) {
                Statement stmt = o.getStatement();
                DDArrayAccessInfo arrayInfo = new DDArrayAccessInfo(
                        (ArrayAccess)o, DDArrayAccessInfo.read_type, loop,stmt);
                addArrayAccess(arrayInfo, loopArrayAccessMap);
            }
        }
    }

    private void addArrayAccess(
            DDArrayAccessInfo info,
            HashMap<Symbol, ArrayList<DDArrayAccessInfo>> loopArrayAccessMap) {
        Symbol arrayName = SymbolTools.getSymbolOf((info.getArrayAccess()));
        if (loopArrayAccessMap.containsKey(arrayName)) {
            (loopArrayAccessMap.get(arrayName)).add(info);
        } else {
            ArrayList<DDArrayAccessInfo> infoList =
                    new ArrayList<DDArrayAccessInfo>();
            infoList.add(info);
            loopArrayAccessMap.put(arrayName, infoList);
        }
    }

    /**
    * runDDTest uses framework information collected by the DDTDriver pass to
    * obtain a pair of array accesses and pass them to the DDTest calling
    * interface.
    */
    @SuppressWarnings("unchecked")
    private DDGraph runDDTest(
            Loop loop,
            HashMap<Symbol, ArrayList<DDArrayAccessInfo>> loopArrayAccessMap,
            HashMap<Loop, LoopInfo> loopInfoMap) {
        boolean depExists = false;
        ArrayList<DependenceVector> DVset = null;
        // Dependence graph to hold direction vectors for current loop nest
        DDGraph loopDDGraph = new DDGraph();
        // Obtain the nest we're currently taking into consideration
        LinkedList<Loop> enclosing_nest =
                LoopTools.calculateInnerLoopNest(loop);
        // Get all the array names from the loopArrayAccessMap
        // For each name, a list of accesses is obtained and a pair of accesses
        // such that at least one is of 'DDArrayAccessInfo.write_type' is
        // passed to DDTest for dependence analysis
        Set<Symbol> arrayNames = loopArrayAccessMap.keySet();
        for (Symbol name : arrayNames) {
            ArrayList<DDArrayAccessInfo> arrayList =
                new ArrayList<DDArrayAccessInfo>(loopArrayAccessMap.get(name));
            ArrayList<DDArrayAccessInfo> arrayList2 =
                new ArrayList<DDArrayAccessInfo>(loopArrayAccessMap.get(name));
            /////////////////////
            // ALIAS SET CHECK //
            /////////////////////
            Set alias_set = new HashSet();
            DFIterator<Statement> loop_iter =
                    new DFIterator<Statement>(loop, Statement.class);
            loop_iter.pruneOn(ExpressionStatement.class);
            loop_iter.pruneOn(DeclarationStatement.class);
            loop_iter.pruneOn(ReturnStatement.class);
            while (loop_iter.hasNext()) {
                Statement stmt = loop_iter.next();
                if (stmt instanceof ExpressionStatement ||
                    stmt instanceof DeclarationStatement ||
                    stmt instanceof ReturnStatement) {
                    Set aliases = alias_analysis.get_alias_set(stmt, name);
                    if (aliases != null) {
                        alias_set.addAll(aliases);
                    }
                }
            }
            // Add annotation for alias information
            if (PrintTools.getVerbosity() >= 3) {
                addAliasesAnnotation((Statement)loop, name, alias_set);
            }
            if ((alias_set != null) && !(alias_set.isEmpty())) {
                if (alias_set.contains("*")) {
                    // Add all other symbols to the second iterator
                    for (Symbol s : loopArrayAccessMap.keySet()) {
                        arrayList2.addAll(loopArrayAccessMap.get(s));
                    }
                } else {
                    Set<Symbol> aliased_symbols = alias_set;
                    for (Symbol s : aliased_symbols) {
                        if (loopArrayAccessMap.containsKey(s)) {
                            ArrayList<DDArrayAccessInfo> set =
                                    loopArrayAccessMap.get(s);
                            for (DDArrayAccessInfo dda : set) {
                                if (!arrayList2.contains(dda)) {
                                    arrayList2.addAll(
                                            loopArrayAccessMap.get(s));
                                }
                            }
                        }
                    }
                }
            }
            int list1_size = arrayList.size(), list2_size = arrayList2.size();
            for (int i = 0; i < list1_size; i++) {
                DDArrayAccessInfo expr1_info = arrayList.get(i);
                if (expr1_info.getAccessType() ==
                        DDArrayAccessInfo.write_type) {
                    for (int j = 0; j < list2_size; j++) {
                        DDArrayAccessInfo expr2_info = arrayList2.get(j);
                        // Iterate over all accesses with the same name or
                        // aliased
                        PrintTools.printlnStatus(2,
                                "Testing the pair: expr1 -", expr1_info,
                                "expr2 -", expr2_info);
                        ////////////////////////////////////////////////////
                        // For the two accesses, obtain the current eligible
                        // enclosing nest we're testing with respect to
                        LinkedList<Loop> common_eligible_nest =
                                new LinkedList<Loop>();
                        // Obtain common nest of enclosing loops
                        LinkedList<Loop> common_nest = LoopTools.getCommonNest(
                                expr1_info.getAccessLoop(),
                                expr2_info.getAccessLoop());
                        // Find the intersection of the common enclosing loops
                        // with the current nest we're testing with respect to
                        for (Loop l : common_nest) {
                            if (enclosing_nest.contains(l)) {
                                common_eligible_nest.add(l);
                            }
                        }
                        // If the two expressions being considered are aliased
                        // (as identified by alias analysis) i.e. they have
                        // different symbols, don't even perform dependence
                        // testing: see ELSE part
                        Symbol s1 = SymbolTools.getSymbolOf(
                                expr1_info.getArrayAccess());
                        Symbol s2 = SymbolTools.getSymbolOf(
                                expr2_info.getArrayAccess());
                        if (s1.equals(s2)) {
                            ///////////////////////////////////////////////////
                            // Substitute range information if possible in test
                            // expressions
                            // New array access expressions
                            Expression e1, e2;
                            Set<Symbol> expr1_symbols =
                                    SymbolTools.getAccessedSymbols(
                                    expr1_info.getArrayAccess());
                            expr1_symbols.remove(SymbolTools.getSymbolOf(
                                    expr1_info.getArrayAccessName()));
                            Set<Symbol> expr2_symbols =
                                    SymbolTools.getAccessedSymbols(
                                    expr2_info.getArrayAccess());
                            expr2_symbols.remove(SymbolTools.getSymbolOf(
                                    expr2_info.getArrayAccessName()));
                            for (Loop sloop : common_eligible_nest) {
                                Symbol index_sym =
                                        LoopTools.getLoopIndexSymbol(sloop);
                                expr1_symbols.remove(index_sym);
                                expr2_symbols.remove(index_sym);
                            }
                            if (!(expr1_symbols.isEmpty()) ||
                                !(expr2_symbols.isEmpty())) {
                                RangeDomain stmt_e1_range = RangeAnalysis.query(
                                        expr1_info.getParentStatement());
                                RangeDomain stmt_e2_range = RangeAnalysis.query(
                                        expr2_info.getParentStatement());
                                e1 = stmt_e1_range.substituteForward(
                                    expr1_info.getArrayAccess(), expr1_symbols);
                                e2 = stmt_e2_range.substituteForward(
                                    expr2_info.getArrayAccess(), expr2_symbols);
                            } else {
                                e1 = expr1_info.getArrayAccess().clone();
                                e2 = expr2_info.getArrayAccess().clone();
                            }
                            // Modify the DDArrayAccessInfo objects before
                            // passing to DDTest
                            expr1_info.setArrayAccess((ArrayAccess)e1);
                            expr2_info.setArrayAccess((ArrayAccess)e2);
                            // Create DDTestWrapper with the common nest for the
                            // two array accesses being tested.
                            // DDTestWrapper handles partitioning of the
                            // subscripts, testing for dependence, and returning
                            // an entire set of dependence vectors for the two
                            // array accesses
                            DDTestWrapper ddt = new DDTestWrapper(
                                    expr1_info, expr2_info,
                                    common_eligible_nest, loopInfoMap);
                            DVset = new ArrayList<DependenceVector>();
                            // Pass pair of accesses to dependence test and
                            // store resulting direction vector set in DVset
                            depExists = ddt.testAccessPair(DVset);
                        }
                        // The expressions are conservatively aliased, we cannot
                        // assume details about aliasing. Hence, conservatively
                        // mark as dependent with reference to all directions
                        // for enclosing loops
                        else {
                            DVset = new ArrayList<DependenceVector>();
                            DependenceVector dv =
                                new DependenceVector(common_eligible_nest);
                            DVset.add(dv);
                            depExists = true;
                        }
                        PrintTools.printlnStatus(2, "Dependence vectors:");
                        for (int k = 0; k < DVset.size(); k++) {
                            DependenceVector dv = DVset.get(k);
                            if (dv.isValid()) {
                                PrintTools.printlnStatus(2,dv.VectorToString());
                            }
                        }
                        // For every direction vector in the set, add a
                        // dependence arc in the loop data dependence graph if
                        // depExists
                        if (depExists == true) {
                            for (int k = 0; k < DVset.size(); k++) { 
                                DependenceVector DV = DVset.get(k);
                                DDGraph.Arc arc = new DDGraph.Arc(
                                        expr1_info, expr2_info, DV);
                                loopDDGraph.addArc(arc);
                            }
                        } else if (depExists == false) {
                            PrintTools.printlnStatus(2,
                                    "No dependence between the pair");
                        }
                    }
                }
            }
        }
        PrintTools.printlnStatus(2, loopDDGraph);
        return loopDDGraph;
    }

    private void addAliasesAnnotation(Statement loop, Symbol s, Set alias_set) {
        String annot_string = "";
        annot_string += "Symbol: " + s.toString() + "\n";
        annot_string += "Alias_set: \n" + alias_set.toString();
        CommentAnnotation comment = new CommentAnnotation(annot_string);
        (loop).annotateBefore(comment);
    }
}
