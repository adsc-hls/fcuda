package cetus.analysis;

import cetus.hir.*;

import java.util.*;

/**
 * This analysis detects array- or pointer-type formal parameters that can be
 * safely expressed in terms of the argument arrays. The result of analysis
 * keeps, for each call site, maps from each formal parameters to their
 * equivalent expression in terms of caller's arrays. The analysis is basically  * based on pattern-matching but gives useful information for inlining and for
 * alias analysis.
 * <pre>
 * void func1(double b[][8][8]) {} // or double (*b)[8][8]
 * void func2(double b[][8]) {}    // or double (*b)[8]
 * void func3(double b[]) {}       // or double *b
 * int main() {
 *   double a[8][8][8];
 *   func1(a);        // or func1(&a[0]);
 *   func2(a[0]);     // or func2(&a[0][0]);
 *   func3(a[0][0]);  // or func3(&a[0][0][0]);
 *   return 0;
 * }
 * Result:
 * func1(a) =       { b=>a }
 * func2(a[0]) =    { b=>a[0] }
 * func3(a[0][0]) = { b=>a[0][0] }
 * </pre>
 * In inliner this information is useful when expressing the array access
 * through "b" in terms of "a", keeping the transformed code as easy-to-read as
 * possible. In alias analysis, together with points-to information, the result
 * of this analysis can compute more precise alias-free relationship between
 * two formal parameters if they are equivalent to different portion of one
 * array in the callers.
 */
public class ArrayParameterAnalysis extends AnalysisPass {

    /** Pass name for this analysis */
    private static final String name = "[ArrayParameter]";

    /** Empty range comparator for a simple symbolic analysis */
    private static final RangeDomain empty_rd = new RangeDomain();

    /** Fixed read-only expression */
    private static final Expression zero = new IntegerLiteral(0);

    /** Available safety options */
    public static enum Option {RANGE_CHECK, EXACT_DIMENSION, ALLOW_OFFSET}

    /** List of expression types avoided in function call arguments */
    private static final Set<Class<? extends Traversable>> avoided_arg_child;

    /** Call graph structure */
    private IPAGraph callgraph;

    /**
    * Result of the analysis; it keeps safe array passing as a map from the
    * formal parameters to their equivalent arguments for every call site.
    * This is declared as static to be reused if reanalyisis is not necessary.
    */
    private static Map<Procedure,
                       Map<FunctionCall,
                           Map<Symbol, Expression>>> param_to_args;

    /**
    * Result of pointer range analysis; it keeps the accessed range of the
    * formal parameters within each procedure.
    * This is declared as static to be reused if reanalyisis is not necessary.
    */
    private static Map<Procedure, Map<Symbol, Expression>> pointer_ranges;

    /**
    * Keeps pointers to scalar variables that can be expressed in a simple
    * form when using the corresponding caller's variable. For now it only keeps
    * a set of parameters that are accessed only with pointer dereference in
    * the called procedure.
    */
    private static Set<Symbol> safe_scalars;

    /**
    * Safety option for the analysis.
    */
    private Set<Option> options;

    /** Corner cases not handled by this analysis */
    static {
        avoided_arg_child = new HashSet<Class<? extends Traversable>>();
        //avoided_arg_child.add(BinaryExpression.class);
        avoided_arg_child.add(CommaExpression.class);
        avoided_arg_child.add(ConditionalExpression.class);
        avoided_arg_child.add(FunctionCall.class);
        avoided_arg_child.add(StatementExpression.class);
        avoided_arg_child.add(Typecast.class);
        param_to_args = null;
        pointer_ranges = null;
        safe_scalars = null;
    }

    /**
    * Constructs a new analyzer with the given program.
    * @param prog the input program.
    */
    public ArrayParameterAnalysis(Program prog, Option... opts) {
        super(prog);
        options = EnumSet.of(Option.EXACT_DIMENSION); // by default.
        if (opts.length > 0)
            options = EnumSet.copyOf(Arrays.asList(opts));
        if (param_to_args != null) { // stored result is valid.
            callgraph = null;
            return;
        }
        callgraph = new IPAGraph(prog);
        initializeData();
    }

    /**
    * Invalidates the analysis result. This method needs to be invoked after
    * applying any IR-changing transformations.
    */
    public static void invalidate() {
        param_to_args = null;
        pointer_ranges = null;
        safe_scalars = null;
    }

    /** Initializes the data structrue */
    private void initializeData() {
        param_to_args =
                new IdentityHashMap<Procedure,
                                    Map<FunctionCall,
                                        Map<Symbol, Expression>>>();
        pointer_ranges =
                new IdentityHashMap<Procedure,
                                    Map<Symbol, Expression>>();
        for (Iterator<IPANode> iter=callgraph.topiterator(); iter.hasNext();) {
            IPANode node = iter.next();
            Procedure procedure = node.getProcedure();
            param_to_args.put(procedure,
                              new IdentityHashMap<FunctionCall,
                                                  Map<Symbol, Expression>>());
            pointer_ranges.put(node.getProcedure(),
                               new HashMap<Symbol,Expression>());
            for (CallSite calling_site : node.getCallingSites()) {
                param_to_args.get(procedure).put(
                        calling_site.getFunctionCall(),
                        new HashMap<Symbol,Expression>());
            }
        }
        safe_scalars = new HashSet<Symbol>();
    }

    /** Prints out the result of dependence checker based on the analysis */
    private void testDep() {
        for (Iterator<IPANode> iter=callgraph.topiterator(); iter.hasNext();) {
            IPANode node = iter.next();
            System.out.println("Examining " + node.getProcedure().getName());
            for (Symbol param1 : node.getParameters()) {
                for (Symbol param2 : node.getParameters()) {
                    if (param1 != param2 &&
                        (SymbolTools.isPointer(param1) ||
                                SymbolTools.isArray(param1)) &&
                        (SymbolTools.isPointer(param2) ||
                                SymbolTools.isArray(param2))) {
                        System.out.println(param1.getSymbolName() + " <=> " +
                              param2.getSymbolName() + " = " +
                              isDisjoint(param1, param2, node.getProcedure()));
                    }
                }
            }
        }
    }

    /** Prints out the result of safe argument-parameter mapping */
    private void testArg() {
        for (Iterator<IPANode> iter=callgraph.topiterator(); iter.hasNext();) {
            IPANode node = iter.next();
            for (CallSite call_site : node.getCallSites()) {
                System.out.println(PrintTools.line_sep + "CallSite: " +
                        call_site.getFunctionCall());
                IPANode callee = call_site.getCallee();
                if (callee == null) {
                    continue;
                }
                for (Symbol param : callee.getParameters()) {
                    if (!SymbolTools.isPointer(param) &&
                        !SymbolTools.isArray(param)) {
                        continue;
                    }
                    System.out.println("  " + param.getSymbolName() + " => " +
                            getCompatibleArgument(call_site.getFunctionCall(),
                            param));
                }
            }
        }
    }

    /** Prints out the result of the analysis */
    private void showResult() {
        for (Procedure procedure : pointer_ranges.keySet()) {
            System.out.println(PrintTools.line_sep + "Pointer Range: " +
                    procedure.getName());
            for (Symbol param : pointer_ranges.get(procedure).keySet()) {
                System.out.println("  " + param.getSymbolName() + " => " +
                        pointer_ranges.get(procedure).get(param));
            }
        }
        for (Procedure procedure : param_to_args.keySet()) {
            for (FunctionCall call : param_to_args.get(procedure).keySet()) {
                System.out.println(PrintTools.line_sep +"Safe Passing: " +call);
                for (Symbol param :
                        param_to_args.get(procedure).get(call).keySet()) {
                    System.out.println("  " + param.getSymbolName() + " => " +
                            param_to_args.get(procedure).get(call).get(param));
                }
            }
        }
    }

    /**
    * Returns the pass name.
    * @return the pass name "[ArrayParameter]".
    */
    @Override
    public String getPassName() {
        return name;
    }

    /**
    * Starts the analysis; it visits each callee and performs the analysis.
    */
    @Override
    public void start() {
        if (callgraph == null) { // stored result is valid.
            return;
        }
        for (Iterator<IPANode> iter=callgraph.topiterator(); iter.hasNext();) {
            processCallee(iter.next());
        }
        if (PrintTools.getVerbosity() >= 3) {
            //showResult();
            testDep();
            testArg();
        }
    }

    /**
    * Checks if the accessed range of the parameter symbol {@code param} in the
    * called procedure is within the expected boundary at the call site.
    * @param arg the equivalent argument to {@code param} at the call site.
    * @param param_range the accessed range of the formal parameter symbol.
    * @return true if {@code param_range} is enclosed by the expected range (the
    * size of array specifier for {@code arg}).
    */
    private boolean isBounded(Expression arg, Expression param_range) {
        // RANGE_CHECK is not required -- assume it's safe.
        if (!options.contains(Option.RANGE_CHECK))
            return true; 
        // null means it was not possible to consider that the parameter is
        // used safely.
        if (param_range == null)
            return false; 
        Symbol arg_symbol = SymbolTools.getSymbolOf(arg);
        // No expected boundary exisits.
        if (arg_symbol == null || arg_symbol.getArraySpecifiers().isEmpty())
            return false; 
        // Consider multiple passing is difficult --> TODO
        if (SymbolTools.isFormal(arg_symbol))
            return false; 
        ArraySpecifier arg_aspec =
                (ArraySpecifier)arg_symbol.getArraySpecifiers().get(0);
        int dimension_to_check = 0;
        if (arg instanceof ArrayAccess)
            dimension_to_check += ((ArrayAccess)arg).getNumIndices();
        Expression dim = arg_aspec.getDimension(dimension_to_check);
        return (dim != null &&
                empty_rd.isLE(zero,
                              RangeExpression.toRange(param_range).getLB()) &&
                empty_rd.isLT(RangeExpression.toRange(param_range).getUB(),
                              dim));
    }

    /**
    * Checks if the two symbols are alias-free between themselves. This assumes
    * that the only source of the alias is from the call site. Hence, this
    * method should always be used with more general alias/pointer analysis.
    */
    public boolean
            isDisjoint(Symbol symbol1, Symbol symbol2, Procedure procedure) {
        if (!SymbolTools.isFormal(symbol1) || !SymbolTools.isFormal(symbol2))
            return false; // only formal parameters are considered.
        Set<Symbol> safe_parameters = pointer_ranges.get(procedure).keySet();
        if (!safe_parameters.contains(symbol1) ||
            !safe_parameters.contains(symbol2))
            return false; // it is not accessed in an expected manner.
        for (FunctionCall fcall : param_to_args.get(procedure).keySet()) {
            Expression e1 =
                    param_to_args.get(procedure).get(fcall).get(symbol1);
            Expression e2 =
                    param_to_args.get(procedure).get(fcall).get(symbol2);
            if (e1 == null || e2 == null ||
                !isBounded(e1, pointer_ranges.get(procedure).get(symbol1)) ||
                !isBounded(e2, pointer_ranges.get(procedure).get(symbol2)) ||
                isDependent(e1, e2))
                return false;
        }
        return true;
    }

    /**
    * Returns caller's expression that is equivalent to the given parameter.
    * Inliner can just copy the return expression of this method (if not null),
    * and insert the all subscripts used with the parameter to the returning
    * expression which is either an identifier or an array access.
    * @param fcall the function call of interest.
    * @param param the formal parameter of interest.
    * @return the compatible expression with param, or null if not compatible.
    */
    public Expression getCompatibleArgument(FunctionCall fcall, Symbol param) {
        Procedure procedure = fcall.getProcedure();
        if (procedure == null)
            return null;
        Set<Symbol> safe_parameters = pointer_ranges.get(procedure).keySet();
        if (!safe_parameters.contains(param) && !safe_scalars.contains(param))
            return null;
        return param_to_args.get(procedure).get(fcall).get(param);
    }

    /**
    * Checks if the two expression have possible data dependence.
    * The symbolic comparison is performed with an empty range domain, so the
    * accuracy is very limited.
    */
    private static boolean isDependent(Expression e1, Expression e2) {
        e1 = Symbolic.simplify(e1);
        e2 = Symbolic.simplify(e2);
        if (e1.equals(e2))
            return true;  // same expression.
        Symbol var1 = SymbolTools.getSymbolOf(e1);
        Symbol var2 = SymbolTools.getSymbolOf(e2);
        if (!var1.equals(var2))
            return false; // base name is different.
        if (SymbolTools.isArray(var1) &&
                e1 instanceof ArrayAccess &&
                e2 instanceof ArrayAccess) {
            ArrayAccess acc1 = (ArrayAccess)e1;
            ArrayAccess acc2 = (ArrayAccess)e2;
            for (int i = 0; i < acc1.getNumIndices() &&
                    i < acc2.getNumIndices(); i++) {
                Relation rel =
                        empty_rd.compare(acc1.getIndex(i), acc2.getIndex(i));
                if (rel.isNE())
                    return false; // most significant indices differ.
            }
        }
        return true;
    }

    /** Checks if the symbol has multiple pointer types. */
    private static boolean isMultiplePointer(Symbol symbol) {
        List type_specifiers = symbol.getTypeSpecifiers();
        return (type_specifiers.indexOf(PointerSpecifier.UNQUALIFIED) !=
                type_specifiers.lastIndexOf(PointerSpecifier.UNQUALIFIED));
    }

    /**
    * Returns the effective dimension of the given pointer symbol. It is assumed
    * the specified symbol is a parameter of pointer type.
    * TODO: watch out for function pointer type.
    */
    private static int getEffectiveDimensions(Symbol symbol) {
        int ret = 0;
        if (symbol instanceof NestedDeclarator ||
            symbol.getTypeSpecifiers().contains(PointerSpecifier.UNQUALIFIED))
            ret++;
        List array_specifiers = symbol.getArraySpecifiers();
        if (!array_specifiers.isEmpty() &&
            array_specifiers.get(0) instanceof ArraySpecifier)
            ret += ((ArraySpecifier)array_specifiers.get(0)).getNumDimensions();
        return ret;
    }

    /**
    * This method visits each uses of pointer parameters and computes the range
    * of the pointers (most significant dimension of the array accesses), while
    * marking any unsafe uses of the pointer parameters. The result is stored
    * in "pointer_ranges" map.
    */
    private void computePointerRanges(IPANode node) {
        Procedure procedure = node.getProcedure();
        Set<Symbol> defined_symbols = DataFlowTools.getDefSymbol(procedure);
        Map<Symbol, Expression> pointer_params = pointer_ranges.get(procedure);
        Set<Symbol> unsafe_params = new HashSet<Symbol>();
        for (Symbol param : node.getParameters()) {
            if ((SymbolTools.isPointer(param) && !isMultiplePointer(param)) ||
                    SymbolTools.isArray(param) )
                // Put a -INF to mark it has not been tested.
                pointer_params.put(param, new InfExpression(-1));
        }
        DepthFirstIterator<Traversable> iter =
                new DepthFirstIterator<Traversable>(procedure.getBody());
        iter.pruneOn(ArrayAccess.class);
        while (iter.hasNext()) {
            Object o = iter.next();
            if (o instanceof ArrayAccess) {
                ArrayAccess acc = (ArrayAccess)o;
                // Only examine the use of parameters.
                if (!IRTools.containsSymbols(acc, pointer_params.keySet()))
                    continue;
                // Quick decision for difficult child types.
                if (IRTools.containsClasses(acc, avoided_arg_child)) {
                    Set<Symbol> acc_symbols =
                            SymbolTools.getAccessedSymbols(acc);
                    acc_symbols.retainAll(pointer_params.keySet());
                    unsafe_params.addAll(acc_symbols);
                    continue;
                }
                Symbol array_symbol =
                        SymbolTools.getSymbolOf(acc.getArrayName());
                // ArrayAccesses of our interest.
                if (acc.getArrayName() instanceof Identifier &&
                        pointer_params.containsKey(array_symbol) &&
                        !unsafe_params.contains(array_symbol) ) {
                    // Effective dimensions should match.
                    if (acc.getNumIndices() ==
                            getEffectiveDimensions(array_symbol)) {
                        RangeDomain domain = empty_rd;
                        if (options.contains(Option.RANGE_CHECK))
                            domain = RangeAnalysis.query(acc.getStatement());
                        Expression range = domain.expandSymbols(
                                acc.getIndex(0), defined_symbols);
                        // No pointer range results in unsafe usage.
                        Expression prev = pointer_params.get(array_symbol);
                        // First visit
                        if (prev instanceof InfExpression) {
                            pointer_params.put(array_symbol, range);
                        }
                        // Otherwise takes union of the ranges.
                        else {
                            range = RangeDomain.unionRanges(
                                    prev, domain, range, domain);
                            pointer_params.put(array_symbol, range);
                        }
                    }
                    // Otherwise, they are not safe.
                    else {
                        unsafe_params.add(array_symbol);
                    }
                // All other types of uses are considered unsafe.
                } else {
                    Set<Symbol> acc_symbols =
                            SymbolTools.getAccessedSymbols(acc);
                    acc_symbols.retainAll(pointer_params.keySet());
                    for (Symbol acc_symbol : acc_symbols)
                        unsafe_params.add(acc_symbol);
                }
            // Consider non-array access as an unsafe access.
            } else if (o instanceof Identifier &&
                    pointer_params.containsKey(((Identifier)o).getSymbol())) {
                unsafe_params.add(((Identifier)o).getSymbol());
            }
        }
        pointer_params.keySet().removeAll(unsafe_params);
    }

    /**
    * Analyzes the called procedure. First, it computes the pointer ranges of
    * parameters and fills in "pointer_ranges" map. Then, it examines each
    * call site to see if every safety condition is satisfied.
    */
    private void processCallee(IPANode node) {
        // Makes conservative decisions for variable argument list.
        if (node.containsVarArg())
            return;
        // Computes pointer_ranges
        try {
            computePointerRanges(node);
        } catch (InternalError ex) {
            ; // just ignores any internal errors.
        }
        for (CallSite calling_site : node.getCallingSites()) {
            Map<Symbol, Expression> equiv =
                    param_to_args.get(node.getProcedure()).get(
                    calling_site.getFunctionCall());
            List<Symbol> params = node.getParameters();
            List<Expression> args = calling_site.getArguments();
            for (int i=0; i<params.size(); i++) {
                Expression equiv_expr =
                        processExpression(params.get(i), args.get(i));
                if (equiv_expr == null) {
                    equiv_expr = processScalar(params.get(i), args.get(i),node);
                }
                if (equiv_expr != null)
                    equiv.put(params.get(i), equiv_expr);
            }
        }
    }

    // A normalizer that catches "very" simple pointer arithmetic.
    private Expression processBinary(BinaryExpression be) {
        Expression ret = be;
        Expression lhs = be.getLHS();
        Expression rhs = be.getRHS();
        BinaryOperator op = be.getOperator();
        if (op == BinaryOperator.ADD) {
            if (isBaseAndOffset(lhs, rhs)) {
                ret = new UnaryExpression(UnaryOperator.ADDRESS_OF,
                        new ArrayAccess(lhs.clone(), rhs.clone()));
            } else if (isBaseAndOffset(rhs, lhs)) {
                ret = new UnaryExpression(UnaryOperator.ADDRESS_OF,
                        new ArrayAccess(rhs.clone(), lhs.clone()));
            }
        } else if (op == BinaryOperator.SUBTRACT) {
            if (isBaseAndOffset(lhs, rhs)) {
                ret = new UnaryExpression(UnaryOperator.ADDRESS_OF,
                        new ArrayAccess(lhs.clone(),
                        new UnaryExpression(UnaryOperator.MINUS, rhs.clone())));
            }
        }
        return ret;
    }

    /**
    * Tests if the lhs/rhs pair represents a pointer arithmetic with base and
    * offset.
    */
    private boolean isBaseAndOffset(Expression lhs, Expression rhs) {
        boolean ret = false;
        if (lhs instanceof Identifier && (rhs instanceof IntegerLiteral ||
            rhs instanceof Identifier)) {
            Symbol lhs_symbol = ((Identifier)lhs).getSymbol();
            boolean lhs_is_array = SymbolTools.isArray(lhs_symbol);
            boolean lhs_is_pointer = SymbolTools.isPointer(lhs_symbol);
            boolean rhs_is_integer = SymbolTools.isInteger(
                    SymbolTools.getExactExpressionType(rhs));
            ret = (lhs_is_array ^ lhs_is_pointer && rhs_is_integer);
        }
        return ret;
    }

    /**
    * Detects a simple pattern that passes the address of a scalar variable to
    * the callee.
    */
    private Expression
            processScalar(Symbol param, Expression arg, IPANode node) {
        Expression ret = null;
        Procedure proc = node.getProcedure();
        if (proc == null ||
            !(arg instanceof UnaryExpression) ||
            !(((UnaryExpression)arg).getExpression() instanceof Identifier))
            return ret;
        Symbol arg_symbol = SymbolTools.getSymbolOf(arg);
        if (!SymbolTools.isPointer(param) ||
            SymbolTools.isPointer(arg_symbol) ||
            !SymbolTools.isScalar(arg_symbol))
            return ret;
        DepthFirstIterator<Traversable> iter =
                new DepthFirstIterator<Traversable>(proc.getBody());
        while (iter.hasNext()) {
            Traversable t = iter.next();
            if (t instanceof Identifier &&
                param.equals(((Identifier)t).getSymbol())) {
                Traversable parent = t.getParent();
                if (parent instanceof UnaryExpression &&
                    ((UnaryExpression)parent).getOperator() ==
                    UnaryOperator.DEREFERENCE) {
                    ret = arg.clone();
                } else {
                    ret = null;
                    break;
                }
            }
        }
        if (ret != null) {
            safe_scalars.add(param);
        }
        return ret;
    }

    /**
    * Analyzes the relationship between the parameter and the argument.
    * @param param the formal parameter symbol.
    * @param arg the actual argument expression.
    * @return the equivalent representation of the formal parameter in terms of
    * the caller's expression. If a returned expression is a comma expression
    * it contains the caller's expression and offset from there.
    */
    private Expression
            processExpression(Symbol param, Expression arg) {
        Expression ret = null;
        // converts a simple pointer arithmetic to an equivalent unary expr.
        if (arg instanceof BinaryExpression) {
            arg = processBinary((BinaryExpression)arg);
        }
        Symbol arg_symbol = SymbolTools.getSymbolOf(arg);
        if (IRTools.containsClasses(arg, avoided_arg_child) ||
                !(arg instanceof Identifier ||
                arg instanceof UnaryExpression ||
                arg instanceof ArrayAccess) ||
                arg_symbol == null ||
                arg_symbol instanceof Procedure ||
                arg_symbol instanceof ProcedureDeclarator ||
                arg_symbol.getArraySpecifiers().isEmpty()) {
            ;  // unable to analyze this case.
        // pass an equivalent array representation; a => &a[0].
        } else if (arg instanceof Identifier) {
            ArrayAccess array =
                    new ArrayAccess(arg.clone(), new IntegerLiteral(0));
            ret = processArray(param, array, -1);
        // normal case.
        } else if (arg instanceof ArrayAccess) {
            ret = processArray(param, (ArrayAccess)arg, 0);
        // with address_of operator.
        } else if (arg instanceof UnaryExpression) {
            Expression expr = ((UnaryExpression)arg).getExpression();
            UnaryOperator op = ((UnaryExpression)arg).getOperator();
            if (op == UnaryOperator.ADDRESS_OF) {
                if (expr instanceof ArrayAccess) {
                    ArrayAccess array = (ArrayAccess)expr;
                    Expression last_index =
                            array.getIndex(array.getNumIndices()-1);
                    if (last_index.equals(zero)) {
                        ret = processArray(param, array, -1);
                    } else if (options.contains(Option.ALLOW_OFFSET)) {
                        ret = processArray(param, array, -1);
                        if (ret != null) {
                            List<Expression> expr_list =
                                    new LinkedList<Expression>();
                            expr_list.add(ret);
                            expr_list.add(last_index.clone());
                            // Base expression and offset expression are
                            // returned as a comma expression.
                            ret = new CommaExpression(expr_list);
                        }
                    }
                }
            }
        }
        return ret;
    }

    /**
    * Analyzes the relationship between the formal parameter and the array
    * arguments with the specified offset.
    * @param param the formal parameter symbol.
    * @param arg the actual array argument.
    * @param offset the offset to indicate the 
    */
    private Expression processArray(Symbol param, ArrayAccess arg, int offset) {
        ArraySpecifier arg_spec = (ArraySpecifier)SymbolTools.getSymbolOf(arg).
                getArraySpecifiers().get(0);
        List param_specs = param.getArraySpecifiers();
        List<Expression> param_dimensions = new LinkedList<Expression>();
        // Normalize parameter's array specifier.
        param_dimensions.add(null);
        if (!param_specs.isEmpty()) {
            ArraySpecifier param_spec = (ArraySpecifier)param_specs.get(0);
            int start = 1;
            if (param instanceof NestedDeclarator)
                start = 0;
            for (int i = start; i < param_spec.getNumDimensions(); i++)
                param_dimensions.add(param_spec.getDimension(i));
        }
        // Check if the tailing dimensions match.
        int head_dimensions = arg.getNumIndices()+offset;
        int tail_dimensions = arg_spec.getNumDimensions() - head_dimensions;
        if (tail_dimensions != param_dimensions.size())
            return null; // returns null if exact matching fails.
        if (options.contains(Option.EXACT_DIMENSION)) {
            for ( int i=1; i<tail_dimensions; i++ ) {
                if (!param_dimensions.get(i).equals(
                        arg_spec.getDimension(head_dimensions+i)))
                    return null;
            }
        }
        // Return null if size matching fails.
        else {
            Expression param_size = new IntegerLiteral(1);
            Expression arg_size = param_size;
            for (int i=1; i<tail_dimensions; i++) {
                param_size = Symbolic.multiply(
                        param_size, param_dimensions.get(i));
                arg_size = Symbolic.multiply(
                        arg_size, arg_spec.getDimension(head_dimensions+i));
            }
            if (!param_size.equals(arg_size))
                return null;
        }
        // Checks equivalent array access.
        Expression ret = new Identifier(SymbolTools.getSymbolOf(arg));
        if (head_dimensions > 0) {
            ret = new ArrayAccess(ret, arg.getIndex(0).clone());
            for (int i = 1; i < head_dimensions; i++)
                ((ArrayAccess)ret).addIndex(arg.getIndex(i).clone());
        }
        return ret;
    }
}
