package cetus.transforms;

import cetus.analysis.LoopTools;
import cetus.analysis.RangeAnalysis;
import cetus.analysis.RangeDomain;
import cetus.exec.Driver;
import cetus.hir.*;

import java.util.*;

/* From OpenMP 3.0
Restrictions
The restrictions to the reduction clause are as follows:
- A list item that appears in a reduction clause of a worksharing construct
  must be shared in the parallel regions to which any of the worksharing regions
  arising from the worksharing construct bind.
- A list item that appears in a reduction clause of the innermost enclosing
  worksharing or parallel construct may not be accessed in an explicit task.
- Any number of reduction clauses can be specified on the directive, but a list
  item can appear only once in the reduction clause(s) for that directive.
--- C/C++ ---
- The type of a list item that appears in a reduction clause must be valid for
  the reduction operator.
- Aggregate types (including arrays), pointer types and reference types may not
  appear in a reduction clause.
- A list item that appears in a reduction clause must not be const-qualified.
- The operator specified in a reduction clause cannot be overloaded with respect
  to the list items that appear in that clause.
--- C/C++ ---
*/

/**
 * This trasformation produces OpenMP-conforming codes from the reduction items
 * recognized by {@link cetus.analysis.Reduction}. The recognition algorithm is
 * lenient enough to allow arbitrary types of expressions in the reduction
 * items, but appropriate code generation is required to produce a code that is
 * accepted by the backend (OpenMP) compiler. The transformation assumes that
 * the loop-parallelization pass does not allow nested parallelism.
 * The command-line option {@code -reduction=N} affects the behavior of this
 * transformation as follows:
 * <ul>
 * <li>-reduction=1: Allows only scalar reduction
 * <li>-reduction=2: Generates code for array reduction with private copies
 * <li>-reduction=3: Protects reduction statement with synchronization
 * <li>-reduction=4: Protects reduction statement if code generation fails
 * </ul>
 * Notice that the third and fourth options always result in parallelized
 * loops even with possible data dependences but the resulting code may be
 * inefficient because of the inserted synchronization.
 */
public class ReductionTransform extends TransformPass {

    /**
     * Pass name
     */
    private static final String pass_name = "[ReductionTransform]";

    /**
     * Reduction strategy driven by command-line option
     */
    private static final int SCALAR_REDUCTION = 1;
    private static final int ARRAY_REDUCTION_COPY = 2;
    private static final int ARRAY_REDUCTION_SYNC = 3;
    private static final int ARRAY_REDUCTION_AUTO = 4;
    private static int option;

    /**
     * Additional options controlled by modifying this code
     */
    private static boolean option_dynamic_copy = true;

    /**
     * Set of data types that are valid for addition/subtraction and
     * multiplication (we allow only these types of reductions).
     */
    private static final Set<Specifier> allowed_data_types;

    /**
     * Read-only data
     */
    private static final Expression zero = new IntegerLiteral(0);
    private static final Expression one = new IntegerLiteral(1);
    private static final List<Object> empty_list =
            Collections.unmodifiableList(new ArrayList<Object>(0));

    /**
     * List of for loops to be transformed
     */
    private List<ForLoop> work_list;

    static {
        allowed_data_types = new HashSet<Specifier>();
        allowed_data_types.add(Specifier.SHORT);
        allowed_data_types.add(Specifier.INT);
        allowed_data_types.add(Specifier.LONG);
        allowed_data_types.add(Specifier.SIGNED);
        allowed_data_types.add(Specifier.UNSIGNED);
        allowed_data_types.add(Specifier.FLOAT);
        allowed_data_types.add(Specifier.DOUBLE);
        try {
            option = Integer.parseInt(Driver.getOptionValue("reduction"));
        } catch (Exception ex) {
            option = SCALAR_REDUCTION;
        } finally {
            if (option < 1 || option > 4)
                option = SCALAR_REDUCTION;
        }
    }

    /**
     * Constructs a new reduction transformation pass.
     *
     * @param program the input program.
     */
    public ReductionTransform(Program program) {
        super(program);
        work_list = new LinkedList<ForLoop>();
        //disable_protection = true;
    }

    /**
     * Starts the transformation.
     */
    public void start() {
        DFIterator<Procedure> iter =
                new DFIterator<Procedure>(program, Procedure.class);
        iter.pruneOn(Procedure.class);
        while (iter.hasNext()) {
            transformProcedure(iter.next());
        }
    }

    /**
     * Returns the pass name.
     */
    public String getPassName() {
        return pass_name;
    }

    /**
     * This method drives procedure-level transformation while computing the
     * value ranges for the procedure.
     *
     * @param proc the procedure to be transformed.
     */
    private void transformProcedure(Procedure proc) {
        work_list.clear();
        scheduleReduction(proc);
        for (ForLoop loop : work_list) {
            transformReduction(loop);
        }
    }

    /**
     * Inspects loop nests in the specified procedure and selects loops that
     * need reduction transformation. After this step, subsequent transformation
     * is performed blindly, so this method should guarantee correctness.
     */
    private void scheduleReduction(Procedure proc) {
        DFIterator<ForLoop> iter = new DFIterator<ForLoop>(proc, ForLoop.class);
        iter.pruneOn(ExpressionStatement.class);
        while (iter.hasNext()) {
            ForLoop loop = iter.next();
            if (loop.containsAnnotation(CetusAnnotation.class, "reduction")) {
                if (loop.containsAnnotation(CetusAnnotation.class,"parallel")) {
                    work_list.add(loop);
                } else {
                // It looks more reasonable to remove reduction pragma
                // when it is not useful for any passes.
                    loop.getAnnotation(CetusAnnotation.class, "reduction").
                            toCommentAnnotation();
                }
            }
        }
    }

    /**
     * Collects reduction items from the specified for loop and drives scalar
     * and array reduction transformations.
     *
     * @param loop the for loop to be transformed.
     * @return true if transformation is successful. False return value
     *         forces removal of cetus parallel annotation.
     */
    private void transformReduction(ForLoop loop) {
        CetusAnnotation reduction =
                loop.getAnnotation(CetusAnnotation.class, "reduction");
        if (reduction == null) {
            return;
        }
        // Set of modified variables
        Set<Symbol> mods = DataFlowTools.getDefSymbol(loop);
        Set<Symbol> vars = new HashSet<Symbol>();
        // List of reduction items and their associated operators.
        List<Expression> scalar_exprs = new LinkedList<Expression>();
        List<String> scalar_operators = new LinkedList<String>();
        List<Expression> array_exprs = new LinkedList<Expression>();
        List<String> array_operators = new LinkedList<String>();
        // Access to the reduction items stored as a map.
        Map<String, Set<Expression>> items = reduction.get("reduction");
        // This loop collects reduction items.
        for (String op : items.keySet()) {
            Set<Expression> item_list = items.get(op);
            if (item_list != null) {
                for (Expression e : item_list) {
                    if (e instanceof Identifier) {
                        ; // do nothing for identifier
                    } else if (e instanceof ArrayAccess) {
                        vars.clear();
                        List<Expression> indices =
                                ((ArrayAccess)e).getIndices();
                        for (Expression index : indices) {
                            vars.addAll(SymbolTools.getAccessedSymbols(index));
                        }
                        if (vars.removeAll(mods)) {
                            array_exprs.add(e);
                            array_operators.add(op);
                        } else {
                            scalar_exprs.add(e);
                            scalar_operators.add(op);
                        }
                    } else if (e instanceof AccessExpression) {
                        // it is possible that e contains array access
                        // (ignore for now).
                        scalar_exprs.add(e);
                        scalar_operators.add(op);
                    } else {
                        throw new InternalError(
                                pass_name + " Unknown " + "reduction type");
                    }
                }
            }
        }
        boolean result =transformScalarReduction(
                loop, scalar_exprs, scalar_operators, items, mods);
        if (!result) {
            PrintTools.printlnStatus(1, pass_name, LoopTools.getLoopName(loop),
                    "was not parallelized (dubious scalar reduction)");
            // Converts the reduction pragma to a comment annotation.
            reduction.toCommentAnnotation();
            return;
        }
        // Quickly return if array reduction is disabled.
        if (option == ARRAY_REDUCTION_AUTO || option == ARRAY_REDUCTION_COPY) {
            result = transformArrayReduction(
                    loop, array_exprs, array_operators, items, mods);
        }
        if (!result) {
            PrintTools.printlnStatus(1, pass_name, LoopTools.getLoopName(loop),
                    "was not parallelized (dubious array reduction)");
            // Converts the reduction pragma to a comment annotation.
            reduction.toCommentAnnotation();
            return;
        }
        scalar_exprs.addAll(array_exprs);
        if (scalar_exprs.size() > 0 && option != ARRAY_REDUCTION_SYNC &&
                option != ARRAY_REDUCTION_AUTO) {
            for (CetusAnnotation note :
                    loop.getAnnotations(CetusAnnotation.class)) {
                if (note.get("parallel") != null) {
                    note.detach();
                }
            }
            PrintTools.printlnStatus(1, pass_name, LoopTools.getLoopName(loop),
                    "was not parallelized (with the reduction option)");
            // Converts the reduction pragma to a comment annotation.
            reduction.toCommentAnnotation();
            return;
        }
        // Enclose the remaining items within a crtical section
        // "atomic" works only for scalar variables
        // "parallel" annotation may be removed if ARRAY_REDUCTION_COPY is
        // forced.
        if (loop.containsAnnotation(CetusAnnotation.class, "parallel")) {
            for (Expression e : scalar_exprs) {
                Statement reduction_stmt = e.getStatement();
                CompoundStatement critical_section = new CompoundStatement();
                CompoundStatement parent = IRTools.getAncestorOfType
                        (reduction_stmt, CompoundStatement.class);
                parent.addStatementBefore(reduction_stmt, critical_section);
                reduction_stmt.detach();
                critical_section.addStatement(reduction_stmt);
                critical_section.annotate(new CetusAnnotation("critical", ""));
            }
        }
        // Clean up reduction pragmas.
        for (String op : new HashSet<String>(items.keySet())) {
            items.get(op).removeAll(scalar_exprs);
            if (items.get(op).isEmpty()) {
                items.remove(op);
            }
        }
        if (items.isEmpty()) {
            reduction.detach();
        }
    }

    /**
     * Performs transformation that 1) assigns a new scalar variable and 2)
     * uses that variable as reduction item so that the transformed code comform
     * the OpenMP specification. Successful transformation removes expressions
     * from {@code exprs}, so non-empty {@code exprs} after calling this method
     * means there exist an unsuccessful transformation.
     *
     * @param loop      the for loop to be affected.
     * @param exprs     the reduction items that need this tranformation.
     * @param operators the type of reduction operations for {@code exprs}.
     * @param items     the reduction map stored in the annotation.
     * @param variants  the set of loop-variant symbols.
     * @return true if transformation is successful. Unsuccessful cases
     *         (false) result in removal of cetus parallel annoation.
     */
    @SuppressWarnings("unchecked")
    private boolean transformScalarReduction(ForLoop loop,
            List<Expression> exprs, List<String> operators, Map<String,
            Set<Expression>> items, Set<Symbol> variants) {
        CompoundStatement parent =
                IRTools.getAncestorOfType(loop, CompoundStatement.class);
        // Allocates a copy to allow modification of exprs
        List<Expression> expr_list = new LinkedList<Expression>();
        List<String> op_list = new LinkedList<String>();
        List<List> type_list = new LinkedList<List>();
        // Filters exceptional cases first
        for (int i = 0; i < exprs.size(); i++) {
            Expression e = exprs.get(i);
            String op = operators.get(i);
            // Skip over the item that contains loop-variant variables.
            if (IRTools.containsSymbols(e, variants)) {
                items.get(op).remove(e);
                continue;
            }
            List types = SymbolTools.getExpressionType(e);
            // Removes unnecessary storage-class specifiers;
            while (types.remove(Specifier.EXTERN)) ;
            while (types.remove(Specifier.STATIC)) ;
            if (types.retainAll(allowed_data_types)) {
                items.get(op).remove(e);
                continue;
                // Discard the item since there exist "unallowed" data types.
            }
            // Populates the work list
            expr_list.add(e);
            op_list.add(op);
            type_list.add(types);
        }
        // Quick return for exceptional cases; these cases are handled only
        // if synchronization is enabled.
        if (expr_list.size() != exprs.size() &&
                option != ARRAY_REDUCTION_SYNC &&
                option != ARRAY_REDUCTION_AUTO) {
            for (CetusAnnotation note :
                    loop.getAnnotations(CetusAnnotation.class)) {
                if (note.get("parallel") != null) {
                    note.detach();
                }
            }
            return false;
        }
        // Performs transformation
        for (int i = 0; i < expr_list.size(); i++) {
            Expression e = expr_list.get(i);
            String op = op_list.get(i);
            List types = type_list.get(i);
            // Performs the following steps:
            // 1. Assign a temporary varaible "v" for "e" with the same type
            // 2. Create/Insert an assignment "v = e" before the loop
            // 3. Replace "e" with "v" within the loop and the pragma
            // 4. Create/Insert an assignment "e = v" after the loop
            Identifier id = SymbolTools.getTemp(loop, types, "reduce");
            Statement copyin = new ExpressionStatement(
                    new AssignmentExpression(id.clone(),
                                             AssignmentOperator.NORMAL,
                                             e.clone()));
            Statement copyout = new ExpressionStatement(
                    new AssignmentExpression(e.clone(),
                                             AssignmentOperator.NORMAL,
                                             id.clone()));
            parent.addStatementBefore(loop, copyin);
            parent.addStatementAfter(loop, copyout);
            IRTools.replaceAll(loop, e, id);
            items.get(op).remove(e);
            items.get(op).add(id);
            exprs.remove(e);
            operators.remove(op);
        }
        return true;
    }

    /**
     * Performs transformation for array reduction. Successful transformation
     * removes items in {@code exprs}.
     *
     * @param loop      the for loop to be affected.
     * @param exprs     the reduction items that need this tranformation.
     * @param operators the type of reduction operations for {@code exprs}.
     * @param items     the reduction map stored in the annotation.
     * @param variants  the set of loop-variant symbols.
     * @return true if the transformation is successful.
     */
    private boolean transformArrayReduction(ForLoop loop,
            List<Expression> exprs, List<String> operators, Map<String,
            Set<Expression>> items, Set<Symbol> variants) {
        // Overview: the order is subject to change for efficient
        // implementation
        // 1. Identify the size of private copy
        // 2. Create and insert preamble
        // 3. Modify the reduction statement in the computation loop
        // 4. Create and insert postamble
        // 5. Enclose 2-4 in a parallel region
        // 6. Fix cetus pragmas

        // parent: parent compound statement of the current loop
        CompoundStatement parent =
                IRTools.getAncestorOfType(loop, CompoundStatement.class);
        // parallel_region: container of the transformed parallel region
        CompoundStatement parallel_region = new CompoundStatement();
        parallel_region.annotate(new CetusAnnotation("parallel", ""));
        // preg_marker: start of the parallel_region (for variable creation)
        AnnotationStatement preg_marker = new AnnotationStatement(
                new CommentAnnotation("Reduction Region"));
        parallel_region.addStatement(preg_marker);
        // critical_section: place for postamble
        CompoundStatement critical_section = new CompoundStatement();
        critical_section.annotate(new CetusAnnotation("critical", ""));

        // Use copy of exprs to allow modification of exprs.
        List<Expression> exprs_copy = new LinkedList<Expression>(exprs);
        for (int i = 0; i < exprs_copy.size(); i++) {
            Expression e = exprs_copy.get(i);
            String op = operators.get(i);
            if (!(e instanceof ArrayAccess))
                continue; // unexpected expression type.
            ArrayAccess orig_acc = (ArrayAccess)e;
            // Computes the reduction span covered by the array access
            //   shared_span : span of array indices at each dimension
            //   private_span: size-1 span is dropped (allocation size)
            List<Expression> shared_span =
                    computeReductionSpan(loop, orig_acc, variants);
            if (shared_span == null) {
                continue;
            }
            List<Expression> private_span =
                    new LinkedList<Expression>(shared_span);
            while (private_span.remove(one)) ;
            if (private_span.contains(null)) { // span should be explicit
                continue;
            }
            // Allocates private copy of the array.
            Declaration private_decl =
                    allocatePrivateCopy(loop, e, op, private_span);
            if (private_decl == null) {
                continue;
            }
            Symbol private_array = (Symbol)private_decl.getChildren().get(0);
            parallel_region.addDeclaration(private_decl);
            // Prepares array accesses using the allocatd private copy for
            // the original loop, preamble, and postamble. Note that the empty
            // loop nest for preamble/postamble is created first to allocate a
            // new index variables for those loops.
            List<Expression> private_indices = new LinkedList<Expression>();
            ForLoop postamble =
                    createLoopNest(private_span, preg_marker, private_indices);
            ArrayAccess private_acc = null;
            ArrayAccess private_acc_post = null;
            ArrayAccess orig_acc_post = orig_acc.clone();
            for (int j = 0; j < shared_span.size(); j++) {
                if (!shared_span.get(j).equals(one)) {
                    Expression private_index = orig_acc.getIndex(j).clone();
                    Expression private_index_post = private_indices.remove(0);
                    if (private_acc == null) {
                        private_acc = new ArrayAccess(
                                new Identifier(private_array), private_index);
                        private_acc_post = new ArrayAccess(
                                new Identifier(private_array),
                                private_index_post);
                    } else {
                        private_acc.addIndex(private_index);
                        private_acc_post.addIndex(private_index_post);
                    }
                    orig_acc_post.setIndex(j, private_index_post.clone());
                }
            }
            // Modifies the original loop body
            IRTools.replaceAll(loop, orig_acc, private_acc);
            items.get(op).remove(e);
            // Prepares and inserts a preamble (no initial values are assumed)
            ForLoop preamble = postamble.clone();
            DFIterator<CompoundStatement> iter =
                    new DFIterator<CompoundStatement>(preamble,
                                                      CompoundStatement.class);
            List<CompoundStatement> prebodies = iter.getList();
            prebodies.get(prebodies.size() - 1).addStatement(
                    new ExpressionStatement(
                    new AssignmentExpression(
                    private_acc_post.clone(), AssignmentOperator.NORMAL,
                    new IntegerLiteral(op.equals("+") ? 0 : 1))));
            parallel_region.addStatement(preamble);
            // Prepares and inserts a postamble
            iter = new DFIterator<CompoundStatement>(postamble,
                                                     CompoundStatement.class);
            List<CompoundStatement> postbodies = iter.getList();
            postbodies.get(postbodies.size() - 1).addStatement(
                    new ExpressionStatement(
                    new AssignmentExpression(
                    orig_acc_post, AssignmentOperator.fromString(op + "="),
                    private_acc_post)));
            critical_section.addStatement(postamble);
            exprs.remove(e);
        }
        // Gives up parallelization if ARRAY_REDUCTION_COPY is forced.
        if (exprs.size() != 0 && option == ARRAY_REDUCTION_COPY) {
            for (CetusAnnotation note :
                    loop.getAnnotations(CetusAnnotation.class)) {
                if (note.get("parallel") != null) {
                    note.detach();
                }
            }
            return false;
        }
        // Prepares and inserts the enclosing parallel region if applicable
        if (exprs.size() < exprs_copy.size()) {
            parallel_region.addStatement(critical_section);
            parent.addStatementBefore(loop, parallel_region);
            parent.removeStatement(loop);
            parallel_region.addStatementBefore(critical_section, loop);
            parallel_region.removeStatement(preg_marker);
            for (CetusAnnotation note :
                    loop.getAnnotations(CetusAnnotation.class)) {
                // Keeps only reduction and lastprivate clauses
                if (note.get("reduction") == null &&
                    note.get("lastprivate") == null) {
                    note.detach();
                }
                // Moves private clause to the parallel region
                if (note.get("private") != null) {
                    parallel_region.annotate(note);
                }
            }
            loop.annotate(new CetusAnnotation("for", ""));
        }
        return true;
    }

    /**
     * Allocates private copy of the specified array access and returns the
     * resulting variable declaration.
     */
    @SuppressWarnings("unchecked")
    private Declaration allocatePrivateCopy(ForLoop loop, Expression e,
            String op, List<Expression> span) {
        Declaration ret = null;
        if (!op.equals("+") && !op.equals("*")) {
            return ret;
        }
        List types = SymbolTools.getExpressionType(e);
        // Removes unnecessary storage-class specifiers;
        while (types.remove(Specifier.EXTERN)) ;
        while (types.remove(Specifier.STATIC)) ;
        if (types.retainAll(allowed_data_types)) {
            // Discard the item since there exist "unallowed" data types.
            return ret;
        }
        NameID private_name = SymbolTools.getNewName("reduce", loop);
        if (option_dynamic_copy) {
            Identifier alloc_name = null;
            alloc_name = SymbolTools.getOrphanID("malloc");
            Expression alloc_arg = span.get(0).clone();
            for (int i = 1; i < span.size(); i++) {
                alloc_arg = Symbolic.multiply(alloc_arg, span.get(i));
            }
            alloc_arg = new BinaryExpression(
                    alloc_arg,
                    BinaryOperator.MULTIPLY,
                    new SizeofExpression(types));
            FunctionCall fc = new FunctionCall(alloc_name, alloc_arg);
            Declarator vdecl = new VariableDeclarator(
                    PointerSpecifier.UNQUALIFIED, private_name);
            Declarator tcdecl = new VariableDeclarator(
                    PointerSpecifier.UNQUALIFIED, new NameID(""));
            if (span.size() > 1) { // requires nested declarator.
                ArraySpecifier aspec =
                        new ArraySpecifier(span.subList(1, span.size()));
                List trailspec = Arrays.asList(aspec);
                vdecl = new NestedDeclarator(
                        empty_list, vdecl, null, trailspec);
                tcdecl = new NestedDeclarator(
                        empty_list, tcdecl, null, trailspec);
            }
            List tcspec = new LinkedList(types);
            tcspec.add(tcdecl);
            Typecast tc = new Typecast(tcspec, fc);
            tc.setParens(false);
            vdecl.setInitializer(new Initializer(tc));
            ret = new VariableDeclaration(types, vdecl);
            // Check and insert stdlib.h
            TranslationUnit tu =
                    IRTools.getAncestorOfType(loop, TranslationUnit.class);
            Declaration first_child = (Declaration)tu.getChildren().get(0);
            if (first_child instanceof AnnotationDeclaration &&
                    first_child.toString().equals("#include <stdlib.h>")) {
                ;
            } else {
                tu.addDeclarationBefore(
                        first_child,
                        new AnnotationDeclaration(
                        new CodeAnnotation("#include <stdlib.h>")));
            }
        } else {
            ArraySpecifier aspec = new ArraySpecifier(span);
            ret = new VariableDeclaration(
                    types, new VariableDeclarator(
                    private_name, Arrays.asList(aspec)));
        }
        return ret;
    }

    /**
     * Computes the span covered by the specified reduction item.
     *
     * @param loop the related reduction loop.
     * @param acc  the reduction item to be processed.
     */
    private List<Expression> computeReductionSpan(ForLoop loop,
            ArrayAccess acc, Set<Symbol> variants) {
        ArraySpecifier aspec = null;
        Symbol array_var = null;
        if (acc.getArrayName() instanceof Identifier) {
            array_var = ((Identifier)acc.getArrayName()).getSymbol();
            List aspecs = array_var.getArraySpecifiers();
            if (array_var instanceof VariableDeclarator &&
                    aspecs.size() == 1 &&
                    aspecs.get(0) instanceof ArraySpecifier &&
                    ((ArraySpecifier)aspecs.get(0)).getNumDimensions() ==
                    acc.getNumIndices()) {
                aspec = (ArraySpecifier)aspecs.get(0);
                // found static declaration of array
            }
        }
        if (array_var == null) {
            return null;
        }
        // Return value contains the span in each dimension.
        List<Expression> ret = new LinkedList<Expression>();
        // search for the relevant reduction statement to derive bounds
        Statement reduction_stmt = null;
        DFIterator<Identifier> iter =
                new DFIterator<Identifier>(loop.getBody(), Identifier.class);
        iter.pruneOn(DeclarationStatement.class);
        iter.pruneOn(AnnotationStatement.class);
        while (iter.hasNext()) {
            Identifier array_id = iter.next();
            if (array_id.getSymbol().equals(array_var)) {
                reduction_stmt = array_id.getStatement();
                break;
            }
        }
        RangeDomain rd = RangeAnalysis.query(reduction_stmt);
        RangeDomain outer_rd = RangeAnalysis.query((Statement)loop);
        if (rd == null) {
            rd = new RangeDomain();
        }
        for (int i = 0; i < acc.getNumIndices(); i++) {
            Expression index = acc.getIndex(i);
            Set<Symbol> vars_in_index = SymbolTools.getAccessedSymbols(index);
            vars_in_index.retainAll(variants);
            if (vars_in_index.isEmpty()) { // assign one if there is no variants
                ret.add(one.clone());
                continue;
            }
            RangeExpression span = RangeExpression.toRange(
                    rd.expandSymbols(index, vars_in_index));
            if (outer_rd.isGE(span.getLB(), zero) &&
                    !(span.getUB() instanceof InfExpression)) {
                                        // found tight bound
                //ret.add(Symbolic.subtract(span.getUB(), span.getLB()));
                // TODO: tighter bound handling requires adjusted subscript
                //       handling in the code generation. For now, just takes
                //       the upper bound assuming the lower bound is zero.
                ret.add(Symbolic.add(span.getUB(), one));
                // note: this may exceed the original size of the array but it
                // is still o.k. since it means conservative space allocation.
                // Implication of this is range-based allocation may or may not
                // be more efficient than static allocation but enables
                // reduction even if the array does not have a static
                // declaration.
            } else if (aspec != null) { // found static bound
                ret.add(aspec.getDimension(i));
            } else { // impossible to acquire a valid bound
                return null;
            }
        }
        return ret;
    }

    /**
     * Creates and returns a loop nest with an empty loop body using the
     * specified scope hint {@code where}. New identifiers are assigned and
     * they are returned through {@code ret_indices}.
     *
     * @param span        the size of the iteration space.
     * @param where       the position where the returned loop is inserted.
     * @param ret_indices the new identifiers used as indices in the loop.
     * @return the created for loop.
     */
    private ForLoop createLoopNest(List<Expression> span, Traversable where,
            List<Expression> ret_indices) {
        ForLoop ret = null;
        for (Expression dim : span) {
            Identifier index =
                    SymbolTools.getTemp(where, Specifier.INT, "reduce_span");
            ForLoop floop = new ForLoop(
                    new ExpressionStatement(
                        new AssignmentExpression(
                            index.clone(),
                            AssignmentOperator.NORMAL,
                            new IntegerLiteral(0))),
                    new BinaryExpression(
                        index.clone(),
                        BinaryOperator.COMPARE_LT,
                        dim.clone()),
                    new UnaryExpression(
                        UnaryOperator.POST_INCREMENT,
                        index.clone()),
                    new CompoundStatement());
            if (ret == null) {
                ret = floop;
            } else {
                ((CompoundStatement)ret.getBody()).addStatement(floop);
            }
            ret_indices.add(index);
        }
        return ret;
    }
}
