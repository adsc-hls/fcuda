package cetus.transforms;

import cetus.analysis.LoopTools;
import cetus.analysis.RangeDomain;
import cetus.hir.*;
import java.util.*;

/**
* Transforms loops so they start with a lower bound of 0
* and run to some upper bound with a stride of 1.
*/
public class LoopNormalization extends TransformPass {

    /** Read-only variables */
    private static final Expression zero = new IntegerLiteral(0);
    private static final Expression one = new IntegerLiteral(1);
    private static final String pass_name = "[LoopNormalization]";

    /**
    * Expression types not handled by loop normalizer. The expressions for
    * index, initial value, step value, and last value should not contain any
    * of these expressions.
    */
    private static final Set<Class<? extends Traversable>> avoid_set;
    static {
        avoid_set = new HashSet<Class<? extends Traversable>>();
        avoid_set.add(AccessExpression.class);
        avoid_set.add(Typecast.class);
        avoid_set.add(UnaryExpression.class);
        avoid_set.add(AssignmentExpression.class);
        avoid_set.add(FunctionCall.class);
    }

    /**
    * Constructs a new loop normalization pass with the specified program.
    * @param program the program to be transformed.
    */
    public LoopNormalization(Program program) {
        super(program);
    }

    /* Returns the pass name */
    public String getPassName() {
        return pass_name;
    }

    /* Starts the transformation */
    public void start() {
        DFIterator<ForLoop> iter =
                new DFIterator<ForLoop>(program, ForLoop.class);
        iter.pruneOn(ExpressionStatement.class);
        iter.pruneOn(VariableDeclaration.class);
        iter.pruneOn(DeclarationStatement.class);
        iter.pruneOn(ClassDeclaration.class);
        while (iter.hasNext()) {
            ForLoop loop = iter.next();
            if (LoopTools.isCanonical(loop)) {
                normalizeLoop(loop);
            }
        }
    }

    /**
    * Extracts expressions that describe the iteration space of the specified
    * loop. When successful, the method should return four expressions each of
    * which are index, initial value, last value, and step size.
    * @param loop the loop to be examined.
    * @return the four expressions that describe the iteration space or null if
    * the loop is not eligible for normalization.
    */
    private static List<Expression> collectIterationSpace(ForLoop loop) {
        // The utility methods in analysis/LoopTools is o.k. here except for
        // computing bound expressions.
        List<Expression> ret = new LinkedList<Expression>();
        // Collects index variable.
        Expression e = null;
        if (!isEligibleExpr(e = LoopTools.getIndexVariable(loop))) {
            return null;
        }
        ret.add(e);
        // Collects initial value.
        if (!isEligibleExpr(e = LoopTools.getLowerBoundExpression(loop))) {
            return null;
        }
        ret.add(e);
        // Check condition expression
        e = null;
        BinaryOperator op = null;
        List conditions = Symbolic.getVariablesOnLHS(loop.getCondition());
        for (Object o : conditions) {
            if (!(o instanceof BinaryExpression)) {
                return null;
            }
            BinaryExpression condition = (BinaryExpression)o;
            if (condition.getLHS().equals(ret.get(0))) {
                e = condition.getRHS();
                op = checkEligibleCondition(condition);
                break;
            }
        }
        if (op == null || !isEligibleExpr(e)) {
            return null;
        }
        // Adjust LT/GT
        if (op == BinaryOperator.COMPARE_LT) {
            e = Symbolic.subtract(e, one);
        } else if (op == BinaryOperator.COMPARE_GT) {
            e = Symbolic.add(e, one);
        }
        ret.add(e);
        // Check step expression
        if (!isEligibleExpr(e = LoopTools.getIncrementExpression(loop))) {
            return null;
        }
        if (!(e instanceof IntegerLiteral)) {
            // Invoking range analysis may partially solve this problem, but
            // it is too costly for this purpose.
            PrintTools.printlnStatus(1, pass_name,
                    "Symbolic step expression may evaluate to zero.",
                    LoopTools.toControlString(loop));
            return null;
        }
        ret.add(e);
        return ret;
    }

    /**
    * Checks if the given expression is eligible for transformation by
    * searching for any expression types that should be avoided within the
    * given expression.
    * @param e the expression to be checked for eligibility.
    * @return true if <code>e</code> is eligible, false otherwise.
    */
    private static boolean isEligibleExpr(Expression e) {
        if (e == null) {
            return false;
        }
        // Filter complicated expressions that may incur unsafe operations.
        if (IRTools.containsClasses(e, avoid_set)) {
            return false;
        }
        // Filter non-integer types; it allows only 'int/long' types.
        List specs = SymbolTools.getExpressionType(e);
        if (specs == null) {
            return false;
        }
        // Check if the expression type is simple enough.
        while (specs.remove(Specifier.STATIC));
        //while (specs.remove(Specifier.EXTERN)) ;
        return (specs.size() == 1 &&
            (specs.get(0) == Specifier.INT || specs.get(0) == Specifier.LONG));
    }

    /** Checks if the expression is eligible binary expression */
    private static BinaryOperator checkEligibleCondition(Expression e) {
        if (!(e instanceof BinaryExpression)) {
            return null;
        }
        BinaryOperator bop = ((BinaryExpression)e).getOperator();
        if (bop == BinaryOperator.COMPARE_LT ||
            bop == BinaryOperator.COMPARE_LE ||
            bop == BinaryOperator.COMPARE_GT ||
            bop == BinaryOperator.COMPARE_GE) {
            return bop;
        } else {
            return null;
        }
    }

    /**
    * Performs loop normalization.
    * Algorithm is based on the one in "Optimizing Compilers for Modern
    * Architectures, Allen and Kennedy".
    * <pre>
    *   i  - original index variable
    *   I  - original initial value
    *   L  - original limit value
    *   S  - original step value
    *   i0 - new index variable 
    *   Set initial statement as "i = 0".
    *   Set bounding expression as "i &lt = (L-I+S)/S - 1".
    *   Set step expression as "i++".
    *   Replace the use of "i" with "i0*S + I" within the loop body.
    *   Append last value assignment "i = i0*S + I".
    * </pre>
    */
    private void normalizeLoop(ForLoop loop) {
        List<Expression> space = collectIterationSpace(loop);
        // space[0] => loop index
        // space[1] => initial value
        // space[2] => last value
        // space[3] => step value
        if (space == null) {
            PrintTools.printlnStatus(1, pass_name,
                    "Loop is ineligible for normalization:");
            PrintTools.printlnStatus(1, pass_name, 
                    LoopTools.toControlString(loop));
            return;
        } else if (space.get(1).equals(zero) && space.get(3).equals(one)) {
            PrintTools.printlnStatus(1, pass_name,
                    "Loop is already in normal form:");
            PrintTools.printlnStatus(1, pass_name,
                    LoopTools.toControlString(loop));
            return;
        }
        // Attach comment.
        CommentAnnotation comment = new CommentAnnotation("Normalized Loop");
        comment.setOneLiner(true);
        loop.annotateBefore(comment);
        // Modifies the iteration space.
        Identifier index = SymbolTools.getTemp(
                loop.getParent(), Specifier.INT, space.get(0).toString());
        Statement init_stmt = new ExpressionStatement(
                new AssignmentExpression(
                        index, AssignmentOperator.NORMAL, zero.clone()));
        Expression condition = Symbolic.subtract(space.get(2), space.get(1));
        condition = Symbolic.add(condition, space.get(3));
        condition = Symbolic.divide(condition, space.get(3));
        condition = Symbolic.subtract(condition, one);
        condition = new BinaryExpression(
                index.clone(), BinaryOperator.COMPARE_LE, condition);
        Expression step = new UnaryExpression(
                UnaryOperator.POST_INCREMENT, index.clone());
        loop.setInitialStatement(init_stmt);
        loop.setCondition(condition);
        loop.setStep(step);
        // Modifies the loop body by substituting new expression for the
        // original index.
        Expression subst = Symbolic.multiply(index, space.get(3));
        subst = Symbolic.add(subst, space.get(1));
        IRTools.replaceSymbolIn(loop.getBody(),
                ((Identifier)space.get(0)).getSymbol(), subst);
        // Appends last value assignment.
        Statement last_assign = new ExpressionStatement(
                new AssignmentExpression(
                        space.get(0).clone(), AssignmentOperator.NORMAL,subst));
        CompoundStatement parent = IRTools.getAncestorOfType(
                loop, CompoundStatement.class);
        parent.addStatementAfter(loop, last_assign);
    }
}
