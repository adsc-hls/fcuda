package cetus.transforms;

import cetus.hir.*;

import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
* Transforms a program such that every statement contains
* at most one function call.  In the case of nested calls,
* the innermost calls will be called first.  Temporaries
* are introduced to hold the results of function calls.
* Remember that this normalization does not guarantee the intended program
* structure (single call per statement) since there may be a case that is
* not normalized to provide code correctness.
*/
public class SingleCall extends ProcedureTransformPass {

    private static final String pass_name = "[SingleCall]";

    /** Constructs a new SingleCall transformation pass */
    public SingleCall(Program program) {
        super(program);
    }

    // Check if there is any R/W on the same variable except for the top-level
    // assignments; normalizing such cases may result in semantic changes due to
    // different sequence of data accesses from the original code.
    private static boolean containsDependence(Statement stmt) {
        boolean ret = false;
        if (stmt instanceof ExpressionStatement) {
            Expression e = ((ExpressionStatement)stmt).getExpression();
            if (e instanceof AssignmentExpression) {
                AssignmentExpression ae = (AssignmentExpression)e;
                ret = containsDependence(ae.getLHS()) ||
                      containsDependence(ae.getRHS());
            } else {
                ret = containsDependence(e);
            }
        } else if (stmt instanceof ReturnStatement) {
            Expression e = ((ReturnStatement)stmt).getExpression();
            ret = containsDependence(e);
        }   // The other cases are handled separately.
        return ret;
    }

    // Checks if the specified expression contains write access to an object
    // that may prevent the transformation from hoisting function calls.
    private static boolean containsDependence(Expression e) {
        Set<Expression> mods = DataFlowTools.getDefSet(e);
/* This code is valid only if there is no side effects on global variables.
        Set<Expression> refs = DataFlowTools.getUseSet(e);
        mods.retainAll(refs);
*/
        return !mods.isEmpty();
    }

    // Returns list of function calls in stmt that can be precomputed before
    // stmt. The returned list should contain the function calls in post order
    // for equivalent evaluation.
    @SuppressWarnings("unchecked")
    private static List<FunctionCall>
        getEligibleFunctionCalls(Statement stmt) {
        List<FunctionCall> ret = new LinkedList<FunctionCall>();
        PostOrderIterator iter =
                new PostOrderIterator(stmt, new Class<?>[] {
                        CompoundStatement.class,      // handled separately
                        DeclarationStatement.class,   // won't be handled
                        ConditionalExpression.class,  // won't be handled
                        StatementExpression.class     // won't be handled
                        });
        while (iter.hasNext()) {
            Object o = iter.next();
            if (o instanceof FunctionCall) {
                FunctionCall fc = (FunctionCall) o;
                if (isUnsafe(stmt, fc)) {
                    ret.clear();
                    break;
                } else if (isUnnecessary(stmt, fc)) {
                    continue;
                }
                ret.add(fc);
            }
        }
        if (ret.size() > 0 && containsDependence(stmt)) {
            ret.clear();
        }
        return ret;
    }

    // Test for unsafe logic
    private static boolean isUnsafeLogic(Statement stmt, FunctionCall fc) {
        Traversable t = fc.getParent();
        while (t != stmt) {
            if (t instanceof BinaryExpression) {
                BinaryOperator bop = ((BinaryExpression) t).getOperator();
                if (bop == BinaryOperator.LOGICAL_AND ||
                    bop == BinaryOperator.LOGICAL_OR) {
                    return true;
                }
            }
            t = t.getParent();
        }
        return false;
    }

    // Test for possible unsafe scenario.
    // 1. BinaryOperation that possibly short-circuited
    // 2. Function pointers
    // 3. Within loop-controlling constructs
    // 4. No type information is extracted
    private static boolean isUnsafe(Statement stmt, FunctionCall fc) {
        BinaryExpression be = null;
        List types = null;
        return (isUnsafeLogic(stmt, fc) ||
                !(fc.getName()instanceof Identifier) ||
                (types = getSpecifiers(fc)) == null ||
                (types.size() == 1 && types.get(0) == Specifier.VOID) ||
                IRTools.getAncestorOfType(fc,Statement.class) instanceof Loop ||
                SymbolTools.getNativeSpecifiers(
                        fc, ((Identifier)fc.getName()).getSymbol()) == null
                // The native type is not extractable.
                );
    }

    // Test for unnecessary transformation.
    // 1. Function call is the only expression in the statement
    // 2. Already in a simple form: lhs = foo();
    private static boolean isUnnecessary(Statement stmt, FunctionCall fc) {
        Traversable parent = fc.getParent();
        return (parent.equals(stmt) ||
                parent.getParent().equals(stmt) &&
                        (parent instanceof AssignmentExpression ||
                parent instanceof Typecast));
    }

    /** Performs transformation for the specified procedure */
    public void transformProcedure(Procedure proc) {
        int num_transforms = 0;
        List<CompoundStatement> comp_stmts =
                IRTools.getDescendentsOfType(proc, CompoundStatement.class);
        String proc_string = proc.toString();
        for (CompoundStatement comp_stmt : comp_stmts) {
            num_transforms += transformCompound(comp_stmt);
        }
        PrintTools.printlnStatus(1, pass_name ,
                "hoisted", num_transforms, "function calls.");
    }

    // Check if it is possible to get valid specifiers from the function call.
    @SuppressWarnings("unchecked")
    private static List getSpecifiers(FunctionCall fc) {
        Symbol symbol = SymbolTools.getSymbolOf(fc.getName());
        if (symbol == null) {
            return null;
        }
        List ret = new LinkedList(symbol.getTypeSpecifiers());
        // Remove specifiers not for types.
        while (ret.remove(Specifier.EXTERN));
        while (ret.remove(Specifier.STATIC));
        while (ret.remove(Specifier.INLINE));
        return ret;
    }

    // Performs transformation for the given compound statement.
    private static int transformCompound(CompoundStatement cs) {
        int ret = 0;
        for (Traversable child : new LinkedList<Traversable>(cs.getChildren())){
            Statement stmt = (Statement) child; // should be guaranteed
            for (FunctionCall fcall : getEligibleFunctionCalls(stmt)) {
                // Actions being performed here for stmt: s = ..... foo() ....;
                // - get a new variable -> foo_0
                // - create a new assignment foo_0 = foo_0;
                // - swap the LHS of the assignment with the original function
                //   call.
                // - inserts the new assignment
                // Result:
                // {
                //   <types> foo_0;
                //   foo_0 = foo();
                //   s = ..... foo_0 .....;
                // }
                List types = getSpecifiers(fcall);
                Identifier id = SymbolTools.getTemp(
                        stmt, types, fcall.getName().toString());
                Statement assign = new ExpressionStatement(
                        new AssignmentExpression(id.clone(),
                                                 AssignmentOperator.NORMAL,
                                                 id));
                id.swapWith(fcall);
                cs.addStatementBefore(stmt, assign);
                CommentAnnotation info =
                        new CommentAnnotation("Normalized Call: " + fcall);
                info.setOneLiner(true);
                assign.annotateBefore(info);
                ret++;
            }
        }
        return ret;
    }

    public String getPassName() {
        return pass_name;
    }

}
