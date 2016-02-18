package cetus.hir;

import java.util.ArrayList;
import java.util.List;

/**
* Class Symbolic provides static utility methods that simplifies expressions
* symbolically. This class holds only wrapper methods that utilizes the
* normalization functionalities in SimpleExpression.
*/
public class Symbolic extends SimpleExpression {

    private static final IntegerLiteral zero = new IntegerLiteral(0);

    private static final IntegerLiteral one = new IntegerLiteral(1);

    // No instantiation is used.
    private Symbolic() {
    }

    /**
    * Returns a simplified expression of the given expression with the default
    * option which turns on every suboptions.
    * @param e the given expression.
    * @return the simplified expression.
    */
    public static Expression simplify(Expression e) {
        SimpleExpression se = new SimpleExpression(e);
        Expression ret = se.normalize().getExpression();
        ret.setParens(e.needsParens());
        return ret;
    }

    /**
    * Returns a simplified expression of the given expression with the
    * user-specified option.
    * @param e the given expression.
    * @param opt the composable simplification option.
    *   e.g., Symbolic.FOLD, Symbolic.FOLD+Symbolic.FACTORIZE, ...
    * @return the simplified expression.
    */
    public static Expression simplify(Expression e, int opt) {
        int orig_option = getOption();
        setOption(opt);
        Expression ret = simplify(e);
        setOption(orig_option);
        return ret;
    }

    /**
    * Returns addition of the two expressions with simplification.
    */
    public static Expression add(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.ADD, e2);
    }

    /**
    * Returns subtraction of two expressions with simplification.
    */
    public static Expression subtract(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.SUBTRACT, e2);
    }

    /**
    * Returns multiplication of two expressions with simplification.
    */
    public static Expression multiply(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.MULTIPLY, e2);
    }

    /**
    * Returns division of two expressions with simplification.
    */
    public static Expression divide(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.DIVIDE, e2);
    }

    /**
    * Returns modulus of two expressions with simplification.
    */
    public static Expression mod(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.MODULUS, e2);
    }

    /**
    * Returns and of two expressions with simplification.
    */
    public static Expression and(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.LOGICAL_AND, e2);
    }

    /**
    * Returns or of two expressions with simplification.
    */
    public static Expression or(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.LOGICAL_OR, e2);
    }

    /**
    * Returns comparison(==) of two expressions with simplification.
    */
    public static Expression eq(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.COMPARE_EQ, e2);
    }

    /**
    * Returns comparison(not equal) of two expressions with simplification.
    */
    public static Expression ne(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.COMPARE_NE, e2);
    }

    /**
    * Returns comparison(less equal) of two expressions with simplification.
    */
    public static Expression le(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.COMPARE_LE, e2);
    }

    /**
    * Returns comparison(less) of two expressions with simplification.
    */
    public static Expression lt(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.COMPARE_LT, e2);
    }

    /**
    * Returns comparison(greater equal) of two expressions with simplification.
    */
    public static Expression ge(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.COMPARE_GE, e2);
    }

    /**
    * Returns comparison(greater) of two expressions with simplification.
    */
    public static Expression gt(Expression e1, Expression e2) {
        return binary(e1, BinaryOperator.COMPARE_GT, e2);
    }

    /**
    * Returns negation of the given expression.
    */
    public static Expression negate(Expression e) {
        return unary(UnaryOperator.LOGICAL_NEGATION, e);
    }

    /**
    * Returns <pre>e</pre>'s power to the <pre>order</pre> with simplification.
    */
    public static Expression power(Expression e, int order) {
        Expression ret = one;
        if (order == 0) {
            ret = ret.clone();
        } else {
            int abs = Math.abs(order);
            for (int i = 0; i < abs; i++) {
                ret = multiply(ret, e);
            }
            if (order < 0) {
                ret = divide(one, ret);
            }
        }
        return ret;
    }

    /**
    * Returns the given expression incremented by one.
    */
    public static Expression increment(Expression e) {
        return add(e, one);
    }

    /**
    * Returns the given expression decremented by one.
    */
    public static Expression decrement(Expression e) {
        return subtract(e, one);
    }

    /**
    * Returns a list of comparison expressions equivalent to the given
    * expression. Each element in the returned list has an identifier on the
    * right-hand side.
    * @param e the given comparison expression.
    * @return a list of equivalent/normalized comparison expressions.
    */
    public static List<Object> getVariablesOnLHS(Expression e) {
        List<Object> ret = new ArrayList<Object>(4);
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        if (se.getOP() == AND || se.getOP() == OR) {
            List<SimpleExpression> children = se.getChildren();
            for (int i = 0; i < children.size(); i++) {
                ret.add(getVariablesOnLHS(children.get(i).getExpression()));
            }
        } else if (se.isCompare()) {
            // 1st phase with the original expression.
            // a-b < 0 --> a < b
            ret.addAll(getVariablesOnLHS(se));
            // 2nd phase with an exchanged expression.
            // a-b < 0 --> b > a
            SimpleExpression exchanged =
                new SimpleExpression(subtract(szero, se.getChild(0)),
                                     exchangeOp(se.getOP()),
                                     subtract(szero, se.getChild(1)));
            ret.addAll(getVariablesOnLHS(exchanged));
        }
        return ret;
    }

    // getVariablesOnLHS : assumes comparison expression
    private static List<Expression> getVariablesOnLHS(SimpleExpression se) {
        List<Expression> ret = new ArrayList<Expression>(4);
        if (!se.isCompare()) {
            return ret;
        }
        SimpleExpression lhs = se.getChild(0), rhs = se.getChild(1);
        if (lhs.getOP() == ID) {
            if (!IRTools.containsExpression(rhs.getExpression(),
                                            lhs.getExpression())) {
                ret.add(se.getExpression());
            }
        } else if (lhs.getOP() == ADD) {
            List<SimpleExpression> children = lhs.getChildren();
            for (int i = 0; i < children.size(); i++) {
                SimpleExpression child = children.get(i);
                if (child.getOP() != ID) {
                    continue;
                }
                SimpleExpression new_rhs = add(rhs, subtract(child, lhs));
                SimpleExpression new_se =
                    new SimpleExpression(child, se.getOP(), new_rhs);
                if (!IRTools.containsExpression(new_rhs.getExpression(),
                                                child.getExprRef())) {
                    ret.add(new_se.getExpression());
                }
            }
        }
        return ret;
    }

    /**
    * Returns the constant term in the simplified expression.
    * @param e the expression to be examined.
    * @return the constant term in "long" type.
    */
    public static long getConstantCoefficient(Expression e) {
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        if (se.getOP() == LIT) {
            return se.getValue().longValue();
        } else if (se.getOP() == ADD) {
            List<SimpleExpression> children = se.getChildren();
            for (int i = 0; i < children.size(); i++) {
                SimpleExpression child = children.get(i);
                if (child.getOP() == LIT) {
                    return child.getValue().longValue();
                }
            }
        }
        return 0;
    }

    /**
    * Returns the constant term with respect to the given set of variables.
    * @param e the expression to be examined.
    * @param ids the list of variables (identifiers).
    * @return the symbolic constant term. 
    */
    public static Expression
        getConstantCoefficient(Expression e, List<Identifier> ids) {
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        Expression expr = se.getExpression();
        Expression ret = null;
        if (se.getOP() == ADD) {
            ret = zero;
            List<SimpleExpression> children = se.getChildren();
            for (int i = 0; i < children.size(); i++) {
                ret = add(ret, getConstantCoefficient(
                        children.get(i).getExpression(), ids));
            }
        } else {
            if (IRTools.containsExpressions(expr, ids)) {
                ret = zero.clone();
            } else {
                ret = expr;
            }
        }
        return ret;
    }

    /**
    * Returns the symbolic coefficient of the given identifier in the
    * expression.
    * @param e the expression to be examined.
    * @param id the identifier.
    * @return the symbolic coefficient of the identifier, null if the expression
    * is too complicated.
    */
    public static Expression getCoefficient(Expression e, Identifier id) {
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        Expression expr = se.getExpression();
        Expression ret = null;
        if (expr.equals(id)) {
            ret = one.clone();
        } else if (!IRTools.containsExpression(expr, id)) {
            ret = zero.clone();
        } else if (se.getOP() == MUL) {
            se.getChildren().remove(new SimpleExpression(id));
            if (se.getChildren().size() == 1) {
                se = se.getChild(0);
            }
            ret = se.getExpression();
            if (IRTools.containsExpression(ret, id)) {
                ret = null;
            }
        } else if (se.getOP() == ADD) {
            ret = zero;
            List<SimpleExpression> children = se.getChildren();
            for (int i = 0; i < children.size(); i++) {
                Expression curr = children.get(i).getExpression();
                curr = getCoefficient(curr, id);
                if (curr == null) {
                    ret = null;
                    break;
                }
                ret = add(ret, curr);
            }
        }
        return ret;
    }

    /**
    * Returns the list of coefficient with respect to the given set of
    * identifiers. The last element of the returned list is symbolic constant
    * coefficient.
    * @param e the expression to be examined.
    * @param ids the list of identifiers.
    * @return the list of coefficients.
    */
    public static List<Expression>
        getCoefficient(Expression e, List<Identifier> ids) {
        List<Expression> ret = new ArrayList<Expression>(4);
        for (int i = 0; i < ids.size(); i++) {
            ret.add(getCoefficient(e, ids.get(i)));
        }
        ret.add(getConstantCoefficient(e, ids));
        return ret;
    }

    /**
    * Returns the list of variables if the given expression is affine.
    * This method replaces NormalExpression.getVariableList(e).
    * @param e the expression to be examined.
    * @return the list of variables if e is affine, null otherwise.
    */
    public static List<Identifier> getVariables(Expression e) {
        List<Identifier> ret = new ArrayList<Identifier>(4);
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        switch (se.getOP()) {
        case ID:
            ret.add((Identifier)se.getExpression());
            break;
        case LIT:
            break;
        case MUL:{
                SimpleExpression term = se.getTerm();
                if (term.getOP() == ID) {
                    ret.add((Identifier)term.getExpression());
                } else {
                    ret = null;
                }
                break;
            }
        case ADD:{
                List<SimpleExpression> children = se.getChildren();
                for (int i = 0; i < children.size(); i++) {
                    SimpleExpression term = children.get(i).getTerm();
                    if (term.getOP() == ID) {
                        ret.add((Identifier)term.getExpression());
                    } else if (term.getOP() != LIT) {
                        ret = null;
                        break;
                    }
                }
                break;
            }
        default:
            ret = null;
        }
        return ret;
    }

    /**
    * Checks if the given expression is an affine expression with respect to the
    * given list of identifiers.
    * @param e the expression to be examined.
    * @param ids the list of identifiers.
    * @return true if it is affine, false otherwise.
    */
    public static boolean isAffine(Expression e, List<Identifier> ids) {
        List<Identifier> vars = getVariables(e);
        if (vars == null) {
            return false;
        }
        vars.removeAll(ids);
        return vars.isEmpty();
    }

    /**
    * Returns the closed form of the given summation parameters,
    * sum(e) s.t. 0&lt;lb&lt;=id&lt;=ub. Computation is based on Bernoulli
    * numbers. For now, assume the given constraints holds when this method is
    * called. We can use range information to assure this assumption in the
    * future.
    * @param id the index variable.
    * @param lb the lower bound of id.
    * @param ub the upper bound of id.
    * @param e the given expression.
    * @return the closed form of the given expression.
    */
    public static Expression getClosedFormSum(Identifier id, Expression lb,
                                              Expression ub, Expression e) {
        Expression ret = null;
        if (!lb.equals(one)) {
            Expression new_id = subtract(add(id, lb), one);
            ub = add(subtract(ub, lb), one);
            e = Symbolic.simplify(
                    IRTools.replaceSymbol(e, id.getSymbol(), new_id));
        }
        List<Expression> poly = getPolynomialCoef(e, id);
        if (poly != null) {
            ret = zero;
            for (int i = 0; i < poly.size(); i++) {
                Expression temp = getSumTemplate(ub, i, poly.get(i));
                if (temp == null) {
                    ret = null;
                    break;
                }
                ret = add(ret, temp);
            }
        }
        return ret;
    }

    /**
    * Returns a simplified expression assuming the given expression is
    * divisible.
    */
    public static Expression simplifyDivisible(Expression e) {
        SimpleExpression se = new SimpleExpression(e);
        se = se.normalize().normalizeDivisible();
        Expression ret = se.getExpression();
        return ret;
    }

    /**
    * Computes and returns the closed form expression of the given expression,
    * index, and bounds with the divisibility property of the given expression.
    */
    public static Expression getClosedFormSum(Identifier id, Expression lb,
                                              Expression ub, Expression e,
                                              Boolean divisible) {
        Expression ret = getClosedFormSum(id, lb, ub, e);
        if (ret != null || !divisible) {
            return ret;
        }
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        if (se.getOP() == DIV) {
            // Compute the sum of dividend and divide it with the divider.
            SimpleExpression dividend = se.getChildren().get(0);
            SimpleExpression divider = se.getChildren().get(1);
            if (IRTools.containsSymbol(
                        divider.getExpression(), id.getSymbol())) {
                return null;
            }
            Expression dividend_sum =
                    getClosedFormSum(id, lb, ub, dividend.getExpression());
            if (dividend_sum == null) {
                return null;
            }
            SimpleExpression sum = new SimpleExpression(dividend_sum);
            if (sum.getOP() == DIV) {
                sum = divide(sum.getChildren().get(0),
                             multiply(sum.getChildren().get(1), divider));
            } else {
                sum = divide(sum, divider);
            }
            ret = sum.getExpression();
        } else if (se.getOP() == ADD || se.getOP() == MUL) {
            // Convert it to division expression.
            SimpleExpression div = se.toDivision();
            if (div.getOP() == DIV) {
                ret = getClosedFormSum(
                        id, lb, ub, div.getExpression(), divisible);
            } else {
                ret = getClosedFormSum(id, lb, ub, e);
            }
        }
        return ret;
    }

    /**
    * Returns a list of expressions which contains coefficients of n-th terms
    * when the given id is the basis. e.g.) (2*i*i+1-i,i) returns [1, -1, 2].
    * @param e the input expression.
    * @param id the basis variable.
    * @return the list of coefficients.
    */
    public static List<Expression>
        getPolynomialCoef(Expression e, Identifier id) {
        List<Expression> ret = new ArrayList<Expression>(4);
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        switch (se.getOP()) {
        case ID:
            if (e.equals(id)) {
                ret.add(zero.clone());
                ret.add(one.clone());
            } else {
                ret.add(e.clone());
            }
            break;
        case LIT:
            ret.add(e.clone());
            break;
        case MUL:{
                int skipped_order = 0;
                Expression coef = one;
                List<SimpleExpression> children = se.getChildren();
                for (int i = 0; i < children.size(); i++) {
                    Expression child_expr =
                        children.get(i).getExpression();
                    if (child_expr.equals(id)) {
                        skipped_order++;
                    } else if (IRTools.containsExpression(child_expr, id)) {
                        ret = null;
                        break;
                    } else {
                        coef = multiply(coef, child_expr);
                    }
                }
                if (ret != null) {
                    for (int i = 0; i < skipped_order; i++) {
                        ret.add(zero.clone());
                    }
                    ret.add(coef);
                }
                break;
            }
        case ADD:{
                List<SimpleExpression> children = se.getChildren();
                for (int j = 0; j < children.size(); j++) {
                    Expression child_expr =
                        children.get(j).getExpression();
                    List<Expression> poly =
                        getPolynomialCoef(child_expr, id);
                    if (poly == null) {
                        ret = null;
                        break;
                    } else if (ret.isEmpty()) {
                        ret.addAll(poly);
                    } else {
                        int i = 0;
                        for (; i < ret.size() && i < poly.size(); i++) {
                            ret.set(i, add(ret.get(i), poly.get(i)));
                        }
                        for (; i < poly.size(); i++) {
                            ret.add(poly.get(i));
                        }
                    }
                }
                break;
            }
        default:
            if (IRTools.containsExpression(e, id)) {
                ret = null;
            } else {
                ret.add(e.clone());
            }
        }
        //PrintTools.printlnStatus("[POLY] "+e+" on "+id+" = "+ret, 1);
        return ret;
    }

    // A more general method uses Bernoulli Numbers - for now just use template.
    private static Expression
        getSumTemplate(Expression e, int power, Expression multiplier) {
        Expression ret;
        switch (power) {
        case 0:                // sum_1^n(1) = n
            ret = multiply(e, multiplier);
            break;
        case 1:                // sum_1^n(i) = (n^2+n)/2
            ret = add(power(e, 2), e);
            ret = multiply(ret, multiplier);
            ret = divide(ret, new IntegerLiteral(2));
            break;
        case 2:                // sum_1^n(i^2) = (2*n^3+3*n^2+n)/6
            ret = add(e, multiply(new IntegerLiteral(3), power(e, 2)));
            ret = add(ret, multiply(new IntegerLiteral(2), power(e, 3)));
            ret = multiply(ret, multiplier);
            ret = divide(ret, new IntegerLiteral(6));
            break;
        case 3:                // sum_1^n(i^3) = (n^4+2n^3+n^2)/4
            ret = add(power(e, 2),
                      multiply(new IntegerLiteral(2), power(e, 3)));
            ret = add(ret, power(e, 4));
            ret = multiply(ret, multiplier);
            ret = divide(ret, new IntegerLiteral(4));
            break;
        case 4:                // sum_1^n(i^4) = (6*n^5+15*n^4+10*n^3-n)/30
            ret = subtract(multiply(new IntegerLiteral(10), power(e, 3)), e);
            ret = add(ret, multiply(new IntegerLiteral(15), power(e, 4)));
            ret = add(ret, multiply(new IntegerLiteral(6), power(e, 5)));
            ret = multiply(ret, multiplier);
            ret = divide(ret, new IntegerLiteral(30));
            break;
        default:
            ret = null;
        }
        return ret;
    }

    // Common wrapper method for binary operations.
    private static Expression
        binary(Expression e1, BinaryOperator op, Expression e2) {
        Expression e = new BinaryExpression(e1.clone(), op, e2.clone());
        return simplify(e);
    }

    // Common wrapper method for unary operations.
    private static Expression unary(UnaryOperator op, Expression e) {
        Expression expr = new UnaryExpression(op, e.clone());
        return simplify(expr);
    }

    /**
    * Returns a list of terms for expressions with addition.
    * e.g., a+b*c --&gt; {a, b*c}.
    */
    public static List<Expression> getTerms(Expression e) {
        List<Expression> ret = new ArrayList<Expression>(4);
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        if (se.getOP() == ADD) {
            List<SimpleExpression> children = se.getChildren();
            for (int i = 0; i < children.size(); i++) {
                ret.add(children.get(i).getExpression());
            }
        } else {
            ret.add(se.getExpression());
        }
        return ret;
    }

    public static List<Expression> getFactors(Expression e) {
        List<Expression> ret = new ArrayList<Expression>(4);
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        if (se.getOP() == MUL) {
            List<SimpleExpression> children = se.getChildren();
            for (int i = 0; i < children.size(); i++) {
                ret.add(children.get(i).getExpression());
            }
        } else {
            ret.add(se.getExpression());
        }
        return ret;
    }

    /**
    * Returns a list of denominators if the expression contains divisions.
    * e.g., 2/a+b/c --&gt; {a, c}.
    */
    public static List<Expression> getDenominators(Expression e) {
        List<Expression> ret = new ArrayList<Expression>(4);
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        if (se.getOP() == DIV) {
            ret.add(se.getChildren().get(0).getExpression());
        } else if (se.getOP() == ADD || se.getOP() == MUL) {
            List<SimpleExpression> children = se.getChildren();
            for (int i = 0; i < children.size(); i++) {
                SimpleExpression child = children.get(i);
                if (child.getOP() == DIV) {
                    ret.add(child.getChildren().get(0).getExpression());
                }
            }
        }
        return ret;
    }

    /**
    * Returns an expression that adds all expressions in the given expression
    * list.
    */
    public static Expression addAll(List<Expression> exprs) {
        Expression ret = zero;
        for (int i = 0; i < exprs.size(); i++) {
            ret = add(ret, exprs.get(i));
        }
        return ret;
    }

    /**
    * Returns an expression that multiplies all expressions in the given
    * expression list.
    */
    public static Expression multiplyAll(List<Expression> exprs) {
        Expression ret = one;
        for (int i = 0; i < exprs.size(); i++) {
            ret = multiply(ret, exprs.get(i));
        }
        return ret;
    }

    /**
    * Returns an expression multiplied by LCM of the denominators present in
    * the terms in the specified expression.
    */
    public static Expression multiplyByLCM(Expression e) {
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        List<SimpleExpression> result = se.multiplyByLCM();
        Expression ret = result.get(0).getExpression();
        return ret;
    }

    public static Expression getLeastCommonDenominator(Expression e) {
        Expression ret = null;
        SimpleExpression se = (new SimpleExpression(e)).normalize();
        if (se.getOP() == ADD) {
            ret = getLCD(se.getChildren()).getExpression();
        } else {
            ret = one.clone();
        }
        return ret;
    }

    public static void simplifyWithin(Traversable t) {
        int opt = 0xff ^ COMPARE;
        // LOGIC option is disabled to prevent pointer comparison from
        // modification
        DFIterator<Expression> iter =
            new DFIterator<Expression>(t, Expression.class);
        iter.pruneOn(VariableDeclaration.class);
        iter.pruneOn(Expression.class);
        while (iter.hasNext()) {
            Expression e = iter.next();
            e.swapWith(Symbolic.simplify(e, opt));
        }
    }
}
