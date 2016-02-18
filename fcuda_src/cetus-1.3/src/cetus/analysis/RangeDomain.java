package cetus.analysis;

import cetus.exec.Driver;
import cetus.hir.*;

import java.util.*;

/**
 * RangeDomain provides symbolic environment for symbolic expression
 * comparison. An object of this class keeps track of mapping from a symbol
 * to its corresponding expression that represents valid value range of the
 * symbol. It is guaranteed that, at each program point, this mapping is always
 * true. A range domain object is usually created after the range analysis but
 * an empty range domain is still useful as it can compare simple symbolic
 * values such as <code>e</code> and <code>e+1</code>.
 * The result of expression comparison is recorded in a {@link Relation} object
 * and the relation provides check methods that return the relationship encoded
 * in the object.
 * <p>
 * Following example shows the usage of range domain when comparing two
 * expressions.
 * Suppose the range domain <code>rd</code> was computed with range analysis
 * and <code>rd</code> contains the specified value ranges.
 * <pre>
 * Map&lt;Statement, RangeDomain&gt; range_map = RangeAnalysis.getRanges(proc);
 * RangeDomain rd = range_map.get(stmt);
 * // rd = { i=[1:10], j=[i:20] }
 * Relation rel = rd.compare(i, j); 
 * // rel.isLE() will return true since i&lt;=j&lt;=20.
 * // rel.isGT() will return false since i&lt;=j&lt;=20.
 * </pre>
 */
public class RangeDomain implements Cloneable, Domain {
  /*==================================================================
    Data fields
    ==================================================================*/
    // Debug tag
    private static final String tag = "[RangeDomain]";

    // Maximum comparison depth
    private static final int MAX_COMPARE_DEPTH = 512;

    // Size of the comparison cache
    private static final int COMPARISON_CACHE_SIZE = 512;

    // Flag for range accuracy
    private static final int ACCURACY = 1;

    // Global cache for expression comparison
    private static Cache<List<Object>, Relation> global_cache =
            new Cache<List<Object>, Relation>(COMPARISON_CACHE_SIZE);

    // Debug level
    private static int debug =
        Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();

    // Set of symbolic value ranges.
    private LinkedHashMap<Symbol, Expression> ranges;

    // Global comparison depth counter
    private int compare_depth;

    // Read-only integer literals.
    private static final IntegerLiteral neg_one = new IntegerLiteral(-1);
    private static final IntegerLiteral zero = new IntegerLiteral(0);
    private static final IntegerLiteral one = new IntegerLiteral(1);

  /*==================================================================
    Constructors and data access methods
    ==================================================================*/

    /**
    * Constructs an empty range domain.
    */
    public RangeDomain() {
        ranges = new LinkedHashMap<Symbol, Expression>();
    }

    /**
    * Constructs a new range domain with the given range domain {@code other}.
    * Unlike {@link #clone}, this constructor reuses the expression contained
    * in {@code other}.
    * @param other the original range domain
    */
    public RangeDomain(RangeDomain other) {
        this();
        if (other != null) {
            for (Symbol var : other.ranges.keySet()) {
                setRange(var, other.getRange(var));
            }
        }
    }

    /**
    * Returns a clone of the range domain.
    * @return the cloned range domain.
    */
    @Override
    public RangeDomain clone() {
        RangeDomain o = new RangeDomain();
        for (Symbol var : ranges.keySet()) {
            o.setRange(var, getRange(var).clone());
        }
        return o;
    }

    /**
    * Cleans up the fields of RangeDomain.
    */
    public void clear() {
        ranges.clear();
    }

    /**
    * Returns the number of value ranges in the map.
    * @return  the number of value ranges.
    */
    public int size() {
        return ranges.size();
    }

    /**
    * Updates the value range for the specified variable.
    * @param var    the variable whose value range is updated.
    * @param value  the new value range of the variable.
    */
    public void setRange(Symbol var, Expression value) {
        if (isOmega(value)) {
            ranges.remove(var);
        } else {
            ranges.put(var, value);
        }
    }

    /**
    * Returns the value range for the specified variable.
    * @param var   the variable whose value range is asked for.
    * @return      the value range for the variable.
    */
    public Expression getRange(Symbol var) {
        return ranges.get(var);
    }

    /**
    * Removes the value range for the specified variable.
    * @param var    the variable whose value range is being removed.
    */
    public void removeRange(Symbol var) {
        ranges.remove(var);
    }

    /**
    * Removes value ranges containing the specified variable including the range
    * of the variable.
    * @param var the variable whose containers are removed.
    */
    public void removeRangeWith(Symbol var) {
        Iterator<Symbol> iter = ranges.keySet().iterator();
        while (iter.hasNext()) {
            Symbol symbol = iter.next();
            Expression range = ranges.get(symbol);
            if (symbol == var || IRTools.containsSymbol(range, var)) {
                iter.remove();
            }
        }
    }

    /**
    * Removes value ranges containing the specified set of symbols.
    * @param vars the variables whose containers are removed.
    */
    public void removeRangeWith(Set<Symbol> vars) {
        for (Symbol var : vars) {
            removeRangeWith(var);
        }
    }

    /**
    * Returns the set of variables whose value ranges are present.
    * @return the set of variables.
    */
    public Set<Symbol> getSymbols() {
        return ranges.keySet();
    }

    /**
    * Returns string for this range domain.
    * @return string representation of this object.
    */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(80);
        sb.append("[");
        Map<String, Expression> ordered = new TreeMap<String, Expression>();
        for (Symbol var : ranges.keySet()) {
            ordered.put(var.getSymbolName(), getRange(var));
        }
        Iterator<String> var_iter = ordered.keySet().iterator();
        if (var_iter.hasNext()) {
            String var_name = var_iter.next();
            Expression range = ordered.get(var_name);
            if (range instanceof RangeExpression) {
                Expression lb = ((RangeExpression)range).getLB();
                Expression ub = ((RangeExpression)range).getUB();
                if (lb instanceof InfExpression) {
                    sb.append(var_name).append("<=").append(ub);
                } else if (ub instanceof InfExpression) {
                    sb.append(var_name).append(">=").append(lb);
                } else {
                    sb.append(lb).append("<=").append(var_name);
                    sb.append("<=").append(ub);
                }
            } else {
                sb.append(var_name).append("=").append(range);
            }
        }
        while (var_iter.hasNext()) {
            sb.append(", ");
            String var_name = var_iter.next();
            Expression range = ordered.get(var_name);
            if (range instanceof RangeExpression) {
                Expression lb = ((RangeExpression)range).getLB();
                Expression ub = ((RangeExpression)range).getUB();
                if (lb instanceof InfExpression) {
                    sb.append(var_name).append("<=").append(ub);
                } else if (ub instanceof InfExpression) {
                    sb.append(var_name).append(">=").append(lb);
                } else {
                    sb.append(lb).append("<=").append(var_name);
                    sb.append("<=").append(ub);
                }
            } else {
                sb.append(var_name).append("=").append(range);
            }
        }
        sb.append("]");
        return sb.toString();
    }

    /**
    * Converts this range domain to an equivalent logical expression.
    * e.g., {@code a=[-INF,b]} becomes {@code a&lt=b}.
    * @return the equivalent logical expression.
    */
    public Expression toExpression() {
        Expression ret = null;
        Map<Identifier, Expression> ordered =
                new TreeMap<Identifier, Expression >();
        for (Symbol var : ranges.keySet())
            ordered.put(new Identifier(var), getRange(var).clone());
        for (Identifier var : ordered.keySet()) {
            Expression child = null;
            Expression range = ordered.get(var);
            if (range instanceof RangeExpression) {
                Expression lb = ((RangeExpression)range).getLB().clone();
                Expression ub = ((RangeExpression)range).getUB().clone();
                if (lb instanceof InfExpression) {
                    child = new BinaryExpression(
                            var, BinaryOperator.COMPARE_LE, ub);
                } else if (ub instanceof InfExpression) {
                    child = new BinaryExpression(
                            var, BinaryOperator.COMPARE_GE, lb);
                } else {
                    child = new BinaryExpression(
                                new BinaryExpression(
                                    var, BinaryOperator.COMPARE_GE, lb),
                                BinaryOperator.LOGICAL_AND,
                                new BinaryExpression(
                                    var.clone(),BinaryOperator.COMPARE_LE,ub));
                }
            } else {
                child = new BinaryExpression(
                        var, BinaryOperator.COMPARE_EQ, range);
            }
            if (ret == null) {
                ret = child;
            } else {
                ret = new BinaryExpression(
                        ret, BinaryOperator.LOGICAL_AND, child);
            }
        }
        if (ACCURACY > 1) {
            removeMinMax(ret);
        }
        return ret;
    }

  /*==================================================================
    Core methods for comparison algorithm
    ==================================================================*/

    public boolean isGT(Expression e1, Expression e2) {
        return compare(e1, e2).isGT();
    }

    public boolean isGE(Expression e1, Expression e2) {
        return compare(e1, e2).isGE();
    }

    public boolean isLT(Expression e1, Expression e2) {
        return compare(e1, e2).isLT();
    }

    public boolean isLE(Expression e1, Expression e2) {
        return compare(e1, e2).isLE();
    }

    public boolean isEQ(Expression e1, Expression e2) {
        return compare(e1, e2).isEQ();
    }

    /**
    * Returns the relation between the two expressions under the constraints
    * implied by the set of value ranges in the RangeDomain object. For a single
    * call to this method, a new cache is created to speed up the comparison.
    * @param e1 the first expression being compared
    * @param e2 the second expression being compared
    * @return the {@link Relation} that stores the result of comparison
    */
    public Relation compare(Expression e1, Expression e2) {
        Relation ret = null;
        List<Object> key = getSignature(this, e1, e2);
        if ((ret = global_cache.get(key)) != null) {
            PrintTools.printlnStatus(3, tag, e1, ret, e2, "under", this);
            return ret;
        }
        compare_depth = 0;      // Reset the depth counter.
        ret = compareExpressions(e1, e2);
        // Cache the result globally.
        global_cache.put(key, ret);
        compare_depth = 0;
        PrintTools.printlnStatus(3, tag, e1, ret, e2, "under", this);
        return ret;
    }

    /**
    * Compares two expressions symbolically with the given range domain
    * @param e1 first expression
    * @param rd1 first range domain
    * @param e2 second expression
    * @param rd2 second range domain
    * @return {@link Relation} of the two expressions
    */
    public static Relation compare(Expression e1, RangeDomain rd1,
                                   Expression e2, RangeDomain rd2) {
        Relation ret = rd1.compare(e1, e2);
        if (!ret.isUnknown() && rd1 != rd2) {
            ret = Relation.AND(ret, rd2.compare(e1, e2));
        }
        PrintTools.printlnStatus(3, tag, e1, ret, e2, "under", rd1, "and", rd2);
        return ret;
    }

    /**
    * Evaluates the given logical expression using the constraints.
    * @param e the logical expression.
    * @return the evaluation result; -1 for unknown, 0 for false, 1 for true.
    */
    public int evaluateLogic(Expression e) {
        int ret = -1;
        if (e == null) {
            return ret;
        } else if (e instanceof BinaryExpression) {
            BinaryExpression be = (BinaryExpression)e;
            BinaryOperator bop = be.getOperator();
            Expression lhs = be.getLHS();
            Expression rhs = be.getRHS();
            if (bop.isCompare()) {
                Relation rel;
                if (lhs instanceof Literal && rhs instanceof Literal) {
                    rel = compareLiterals((Literal)lhs, (Literal)rhs);
                } else {
                    rel = compare(lhs, rhs);
                }
                if (bop == BinaryOperator.COMPARE_EQ) {
                    if (rel.isEQ()) {
                        ret = 1;
                    } else if (rel.isGT() || rel.isLT() || rel.isNE()) {
                        ret = 0;
                    }
                } else if (bop == BinaryOperator.COMPARE_GE) {
                    if (rel.isGE() || rel.isGT() || rel.isEQ()) {
                        ret = 1;
                    } else if (rel.isLT()) {
                        ret = 0;
                    }
                } else if (bop == BinaryOperator.COMPARE_GT) {
                    if (rel.isGT()) {
                        ret = 1;
                    } else if (rel.isLT() || rel.isLE() || rel.isEQ()) {
                        ret = 0;
                    }
                } else if (bop == BinaryOperator.COMPARE_LE) {
                    if (rel.isLE() || rel.isLT() || rel.isEQ()) {
                        ret = 1;
                    } else if (rel.isGT()) {
                        ret = 0;
                    }
                } else if (bop == BinaryOperator.COMPARE_LT) {
                    if (rel.isLT()) {
                        ret = 1;
                    } else if (rel.isGT() || rel.isGE() || rel.isEQ()) {
                        ret = 0;
                    }
                } else if (bop == BinaryOperator.COMPARE_NE) {
                    if (rel.isNE() || rel.isGT() || rel.isLT()) {
                        ret = 1;
                    } else if (rel.isEQ()) {
                        ret = 0;
                    }
                }
            } else if (bop.isLogical()) {
                int lhs_eval = evaluateLogic(lhs);
                int rhs_eval = evaluateLogic(rhs);
                if (bop == BinaryOperator.LOGICAL_AND) {
                    if (lhs_eval == 1 && rhs_eval == 1) {
                        ret = 1;
                    } else if (lhs_eval == 0 || rhs_eval == 0) {
                        ret = 0;
                    }
                } else { // LOGICAL_OR
                    if (lhs_eval == 1 || rhs_eval == 1) {
                        ret = 1;
                    } else if (lhs_eval == 0 && rhs_eval == 0) {
                        ret = 0;
                    }
                }
            }
        } else if (e instanceof UnaryExpression) {
            UnaryExpression ue = (UnaryExpression)e;
            if (ue.getOperator() == UnaryOperator.LOGICAL_NEGATION) {
                int eval = evaluateLogic(ue.getExpression());
                if (eval == 1) {
                    ret = 0;
                } else if (eval == 0) {
                    ret = 1;
                }
            }
        }
        return ret;
    }
    // Helper methods that aggressively compares literal expressions.
    private Relation compareLiterals(Literal l0, Literal l1) {
        Relation ret = new Relation();
        if (l0 instanceof StringLiteral || l1 instanceof StringLiteral) {
            return ret;
        }
        double d0, d1;
        if (l0 instanceof IntegerLiteral) {
            d0 = ((IntegerLiteral)l0).getValue();
        } else if (l0 instanceof CharLiteral) {
            d0 = ((CharLiteral)l0).getValue();
        } else {
            d0 = ((FloatLiteral)l0).getValue();
        }
        if (l1 instanceof IntegerLiteral) {
            d1 = ((IntegerLiteral)l1).getValue();
        } else if (l1 instanceof CharLiteral) {
            d1 = ((CharLiteral)l1).getValue();
        } else {
            d1 = ((FloatLiteral)l1).getValue();
        }
        if (d0 == d1) {
            ret.setEQ(true);
        } else if (d0 > d1) {
            ret.setGT(true);
        } else {
            ret.setLT(true);
        }
        return ret;
    }

    // Wrapper for comparisons with expressions and integers
    private Relation compareExpressions(Expression e, int num) {
        return compareExpressions(e, new IntegerLiteral(num));
    }

    // Compare two division expressions normalizing them with the same
    // denominators. e.g., a/b vs. c/d becomes (a*d)/(b*d) vs. (c*b)/(d*b).
    // Returns the replaced difference expression if successful, null otherwise.
    private Expression replaceDivisions(Expression e1, Expression e2) {
        if (!(e1 instanceof BinaryExpression && e2 instanceof BinaryExpression))
            return null;
        BinaryExpression be1 = (BinaryExpression)e1;
        BinaryExpression be2 = (BinaryExpression)e2;
        if (be1.getOperator() != BinaryOperator.DIVIDE ||
            be2.getOperator() != BinaryOperator.DIVIDE)
            return null;
        Expression denom1 = be1.getRHS(), denom2 = be2.getRHS();
        Expression nom1 = be1.getLHS(), nom2 = be2.getLHS();
        int sign_of_denom1 = signOf(denom1), sign_of_denom2 = signOf(denom2);
        // Give up unknown signs of denominators
        if (sign_of_denom1 != 1 && sign_of_denom1 != -1 &&
            sign_of_denom2 != 1 && sign_of_denom2 != -1)
            return null;
        // Normalize the denominator; a/b ? a/c --> (a*c)/(b*c) ? (a*b)/(c*b)
        Expression denom = denom1;
        int sign_of_denom = sign_of_denom1;
        if (!denom1.equals(denom2)) {
            nom1 = Symbolic.multiply(nom1, denom2);
            nom2 = Symbolic.multiply(nom2, denom1);
            denom = Symbolic.multiply(denom1, denom2);
            sign_of_denom = sign_of_denom1 * sign_of_denom2;
        }
        // Normalize sign of denominators
        if (sign_of_denom == -1) {
            nom1 = Symbolic.multiply(neg_one, nom1);
            nom2 = Symbolic.multiply(neg_one, nom2);
            denom = Symbolic.multiply(neg_one, denom);
            sign_of_denom = 1;
        }
        Expression ret = replaceForComparison(Symbolic.subtract(nom1, nom2));
        ret = expandDIV(ret, denom);
        return ret;
    }

/*
    // This comparison routine is not necessary for now.
    // Compare two division expressions.
    // Returns null if not eligible.
    private Relation compareDivisions(Expression e1, Expression e2) {
        if (!(e1 instanceof BinaryExpression) ||
            !(e2 instanceof BinaryExpression)) {
            return null;
        }
        BinaryExpression be1 = (BinaryExpression)e1;
        BinaryExpression be2 = (BinaryExpression)e2;
        if (be1.getOperator() != BinaryOperator.DIVIDE ||
            be2.getOperator() != BinaryOperator.DIVIDE) {
            return null;
        }
        Expression denom1 = be1.getRHS(), denom2 = be2.getRHS();
        Expression nom1 = be1.getLHS(), nom2 = be2.getLHS();
        int sign_of_denom1 = signOf(denom1), sign_of_denom2 = signOf(denom2);

        // Give up unknown signs of denominators
        if (sign_of_denom1 != 1 && sign_of_denom1 != -1 &&
            sign_of_denom2 != 1 && sign_of_denom2 != -1) {
            return null;
        }

        // Normalize the denominator; a/b ? a/c --> (a*c)/(b*c) ? (a*b)/(c*b)
        Expression denom = denom1;
        int sign_of_denom = sign_of_denom1;
        if (!denom1.equals(denom2)) {
            nom1 = Symbolic.multiply(nom1, denom2);
            nom2 = Symbolic.multiply(nom2, denom1);
            denom = Symbolic.multiply(denom1, denom2);
            sign_of_denom = sign_of_denom1*sign_of_denom2;
        }
        // Normalize sign of denominators
        if (sign_of_denom == -1) {
            nom1 = Symbolic.multiply(new IntegerLiteral(-1), nom1);
            nom2 = Symbolic.multiply(new IntegerLiteral(-1), nom2);
            denom = Symbolic.multiply(new IntegerLiteral(-1), denom);
            sign_of_denom = 1;
        }
        // b/a ? c/a, a>0.
        // 1. (b-c)<=-a => int(c/a) > int(b/a)
        // 2. (b-c)>=a  => int(b/a) > int(c/a)
        // Otherwise, add =.
        Expression diff = Symbolic.subtract(nom1, nom2);
        diff = replaceForComparison(diff);
        Relation ret = getRelation(diff);
        if (ret.isUnknown()) {
            ret = new Relation();
        } else if ((ret.isLT() || ret.isGT()) && isLT(abs(diff), abs(denom))) {
            // adjustment for integer division; 0/3 ? 2/3 is equal
            ret.setEQ(true);
        }
        return ret;
    }
*/

    // Returns the absolute value of the specified expression.
    private Expression abs(Expression e) {
        Relation rel = compareExpressions(e, 0);
        if (rel.isGE()) {
            return e;
        } else if (rel.isLE()) {
            return Symbolic.subtract(zero, e);
        } else {
            return null;
        }
    }

    // Apply identity X mod Y = X - Y*BOT(X/Y) if Y!=0.
    // Caution is needed C division does not exactly corresponds to the number
    // theory. This conversion assumes the caller of this method is comparing
    // the given expression and zero.
    private Expression removeDivisions(Expression e) {
        Expression ret = e;
        List<Expression> denoms = Symbolic.getDenominators(e);
        if (!denoms.isEmpty())
            ret = Symbolic.multiplyByLCM(e);
        return ret;
    }

    // Recursive method that compares the two expressions.
    private Relation compareExpressions(Expression e1, Expression e2) {
        Relation ret = new Relation();
        if (compare_depth++ > MAX_COMPARE_DEPTH) {
            return ret;
        }
        // Take difference between two expressions
        Expression diff = diffInfExpressions(e1, e2);
        if (diff == null)
            diff = diffRangeExpressions(e1, e2);
        if (diff == null)
            diff = diffMinMaxExpressions(e1, e2);
        if (diff == null)
            diff = Symbolic.subtract(e1, e2);
        diff = removeDivisions(diff);
        if (diff instanceof IntegerLiteral)
            return getRelation(diff);
        Expression replaced = replaceForComparison(diff);
        if (!isOmega(replaced))
            ret = getRelation(replaced);
        return ret;
    }

    // Expands the given expression
    private Expression replaceForComparison(Expression e) {
        Expression ret = e;
        List<Symbol> order = getReplaceOrder(ret);
        Iterator<Symbol> iter = order.iterator();
        while (iter.hasNext() && !isDecidable(ret)) {
            Symbol var = iter.next();
            ret = replaceSymbol(ret, var, getRange(var));
        }
        // Replace the whole expression with omega
        if (getRelation(ret).isUnknown())
            ret = RangeExpression.getOmega();
        return ret;
    }

    // Decision criteria for stopping successive replacement; it stops right
    // after finding the sign of the expression since difference is always
    // passed to this method.
    private static boolean isDecidable(Expression e) {
        if (e instanceof IntegerLiteral) {
            return true;
        } else if (e instanceof RangeExpression) {
            RangeExpression re = (RangeExpression)e;
            Expression lb = re.getLB(), ub = re.getUB();
            return ((lb instanceof IntegerLiteral
                     || lb instanceof InfExpression)
                    && (ub instanceof IntegerLiteral
                        || ub instanceof InfExpression)
                    || lb instanceof IntegerLiteral
                    && ((IntegerLiteral)lb).getValue() > 0
                    || lb instanceof MinMaxExpression
                    && ((MinMaxExpression)lb).isPosMax()
                    || ub instanceof IntegerLiteral
                    && ((IntegerLiteral)ub).getValue() < 0
                    || ub instanceof MinMaxExpression
                    && ((MinMaxExpression)ub).isNegMin());
        } else {
            return false;
        }
    }

    // Returns equality/inequality of a symbolic comparison
    private static Relation getRelation(Expression e) {
        Relation ret = new Relation();
        if (e instanceof IntegerLiteral) { // Integer literal
            long value = ((IntegerLiteral)e).getValue();
            ret.setLT(value < 0);
            ret.setGT(value > 0);
            ret.setEQ(value == 0);
        } else if (e instanceof RangeExpression) { // Range expression
            RangeExpression re = (RangeExpression) e;
            Expression lb = re.getLB(), ub = re.getUB();
            long lbval = Long.MIN_VALUE, ubval = Long.MAX_VALUE;
            if (lb instanceof IntegerLiteral) {
                lbval = ((IntegerLiteral)lb).getValue();
            } else if (lb instanceof MinMaxExpression &&
                     ((MinMaxExpression) lb).isPosMax()) {
                lbval = 1;
            }
            if (ub instanceof IntegerLiteral) {
                ubval = ((IntegerLiteral)ub).getValue();
            } else if (ub instanceof MinMaxExpression &&
                     ((MinMaxExpression)ub).isNegMin()) {
                ubval = -1;
            }
            if (lbval > ubval) {
                ;
            } else if (lbval < 0) {
                ret.setLT(true);
                ret.setEQ(ubval >= 0);
                ret.setGT(ubval > 0);
            } else if (lbval == 0) {
                ret.setEQ(true);
                ret.setGT(ubval > 0);
            } else
                ret.setGT(true);
        } else if (e instanceof MinMaxExpression) { // MIN/MAX expression
            Long min = null, max = null;
            for (Traversable o : e.getChildren()) {
                if (!(o instanceof IntegerLiteral))
                    continue;
                long value = ((IntegerLiteral)o).getValue();
                if (min == null) {
                    min = new Long(value);
                    max = new Long(value);
                } else {
                    if (value < min)
                        min = value;
                    if (value > max)
                        max = value;
                }
            }
            if (min != null) {
                if (((MinMaxExpression)e).isMin()) {
                    ret.setLT(true);
                    ret.setEQ(min == 0);
                    ret.setGT(min > 0);
                } else {
                    ret.setGT(true);
                    ret.setEQ(max == 0);
                    ret.setLT(max < 0);
                }
            }
        }
        return ret;
    }

    // Wrapper to get the sign of an expression
    private int signOf(Expression e) {
        if (e instanceof InfExpression)
            return ((InfExpression)e).sign();
        Relation rel = compareExpressions(e, 0);
        if (rel.isGT())
            return 1;
        else if (rel.isLT())
            return -1;
        else if (rel.isEQ())
            return 0;
        else
            return 999;
    }

    // Return the difference of two expressions that contain InfExpressions
    private Expression diffInfExpressions(Expression e1, Expression e2) {
        if (e1 instanceof InfExpression) {
            if (e2 instanceof InfExpression)
                return new IntegerLiteral(signOf(e1) - signOf(e2));
            else
                return new IntegerLiteral(signOf(e1));
        } else if (e2 instanceof InfExpression) {
            return new IntegerLiteral(-signOf(e2));
        } else {
            return null;
        }
    }

    // Return the difference of two expressions that contain RangeExpressions
    private Expression diffRangeExpressions(Expression e1, Expression e2) {
        if (e1 instanceof RangeExpression) {
            RangeExpression re1 = (RangeExpression) e1;
            Expression lb = null, ub = null;
            if (e2 instanceof RangeExpression) {
                lb = Symbolic.subtract(re1.getLB(),
                                       ((RangeExpression)e2).getUB());
                ub = Symbolic.subtract(re1.getUB(),
                                       ((RangeExpression)e2).getLB());
            } else {
                lb = Symbolic.subtract(re1.getLB(), e2);
                ub = Symbolic.subtract(re1.getUB(), e2);
            }
            return new RangeExpression(lb, ub);
        } else if (e2 instanceof RangeExpression) {
            Expression lb =
                Symbolic.subtract(e1, ((RangeExpression)e2).getUB());
            Expression ub =
                Symbolic.subtract(e1, ((RangeExpression)e2).getLB());
            return new RangeExpression(lb, ub);
        } else {
            return null;
        }
    }

    // Return the difference of two expressions that contain MinMax.
    private Expression diffMinMaxExpressions(Expression e1, Expression e2) {
        if (!(e2 instanceof MinMaxExpression))
            return null;
        if (e1.equals(e2))
            return new IntegerLiteral(0);
        return Symbolic.add(e1, ((MinMaxExpression)e2).negate());
    }

  /*==================================================================
    Supporting methods for expression manipulation
    ==================================================================*/

    /**
    * Returns the forward-substituted value of a variable from the ranges.
    * The return value is null if there is any dependence cycle in the range
    * dependence graph.
    * @param var the variable whose value is asked for
    * @return the value range of the variable
    */
    protected Expression getForwardRange(Symbol var) {
        List<Symbol> replace_order = getReplaceOrder(getRange(var));
        // Check any duplicates and return null if so (cycles)
        Set<Symbol> vars = new LinkedHashSet<Symbol>();
        for (Symbol key : replace_order) {
            if (vars.contains(key))
                return null;
            vars.add(key);
        }
        Expression ret = getRange(var).clone();
        for (Symbol key : replace_order) {
            ret = replaceSymbol(ret, key, getRange(key));
        }
        return ret;
    }

    // Get symbolic value of var; var must exist in the map.
    private Expression getForwardExpression(Symbol var) {
        Expression ret = getRange(var);
        if (ret instanceof RangeExpression ||
            ret instanceof MinMaxExpression ||
            ret instanceof InfExpression) {
            return null;
        }
        ret = ret.clone();
        Set<Symbol> vars = new LinkedHashSet<Symbol>();
        List<Symbol> replace_order = getReplaceOrder(ret);
        for (Symbol key : replace_order) {
            if (vars.contains(key))
                return null;
            vars.add(var);
            Expression range = getRange(key);
            if (range instanceof RangeExpression ||
                range instanceof MinMaxExpression ||
                range instanceof InfExpression) {
                return ret;
            }
            ret = replaceSymbol(ret, key, range);
        }
        return ret;
    }

    /**
    * Substitute all occurrences of the given symbol in the expression with its
    * symbolic values. The substitution is performed only if there is a
    * non-cyclic substitution order which produces a valid forward substitution.
    * @param e the expression to be substituted.
    * @param var the symbol to be replaced.
    * @return the forward-substituted expression.
    */
    public Expression substituteForward(Expression e, Symbol var) {
        if (getRange(var) == null)
            return e;
        Expression subst = getForwardExpression(var);
        if (subst == null ||
            (subst instanceof Identifier &&
                    ((Identifier)subst).getSymbol() == var) ||
            subst instanceof RangeExpression ||
            subst instanceof MinMaxExpression ||
            subst instanceof InfExpression) {
            return e;
        } else {
            return replaceSymbol(e, var, subst);
        }
    }

    /**
    * Substitute all occrrences of the given set of symbols in the expression.
    * @param e the expression to be substituted.
    * @param vars the set of symbols to be replaced.
    * @return the forward-substituted expression.
    */
    public Expression substituteForward(Expression e, Set<Symbol> vars) {
        if (e == null || vars == null)
            return null;
        Expression ret = e.clone();
        for (Symbol var : vars)
            ret = substituteForward(ret, var);
        return ret;
    }

    /**
    * Substitute all occurrences of the symbols that appear in the expression.
    * @param e the expression to be substituted.
    * @return the resulting expression.
    */
    public Expression substituteForward(Expression e) {
        Set<Symbol> vars = SymbolTools.getAccessedSymbols(e);
        return substituteForward(e, vars);
    }

    /**
    * Applies {@link #substituteForward(Expression)} to the range expressions
    * kept in the range domain.
    */
    public void substituteForward() {
        for (Symbol var : new HashSet<Symbol>(getSymbols()))
            setRange(var, substituteForward(getRange(var)));
    }

    /**
    * Applies {@link #getForwardRange(Symbol)} to the range expressions
    * kept in the range domain.
    */
    public void substituteForwardRange() {
        for (Symbol var : new HashSet<Symbol>(getSymbols()))
            setRange(var, getForwardRange(var));
    }

    // Remove any ranges with a self cycle, e.g., a=[0,a]
    protected void removeRecurrence() {
        for (Symbol var : new LinkedHashSet<Symbol>(ranges.keySet())) {
            if (IRTools.containsSymbol(getRange(var), var))
                removeRange(var);
        }
    }

    // Remove any ranges containing the specified symbol as part of array
    // subscripts.
    protected void killArraysWith(Symbol symbol) {
        for (Symbol var : new LinkedHashSet<Symbol>(ranges.keySet())) {
            Expression range = getRange(var);
            if (range == null)
                continue;
            DFIterator<ArrayAccess> iter =
                    new DFIterator<ArrayAccess>(range, ArrayAccess.class);
            while (iter.hasNext()) {
                for (Expression e : iter.next().getIndices()) {
                    if (IRTools.containsSymbol(e, symbol)) {
                        removeRange(var);
                        break;
                    }
                }
            }
        }
    }

    /**
    * Removes the ranges for the speicfied variables in the map by replacing
    * them with their value ranges.
    * @param vars the set of variables being removed
    */
    public void removeSymbols(Set<Symbol> vars) {
        for (Symbol var : vars) {
            removeSymbol(var);
        }
    }

    /**
    * Replaces all occurrences of the specified variable in the map with the
    * given expression.
    * @param var the variable being replaced
    * @param with the expression being replaced with
    */
    public void replaceSymbol(Symbol var, Expression with) {
        for (Symbol key : new LinkedHashSet<Symbol>(ranges.keySet())) {
            setRange(key, replaceSymbol(getRange(key).clone(), var, with));
        }
    }

    /**
    * Remove a variable in the range domain 
    */
    private void removeSymbol(Symbol var) {
        Expression with = getRange(var);
        removeRange(var);
        for (Symbol symbol : new LinkedHashSet<Symbol>(ranges.keySet())) {
            Expression replaced =
                replaceSymbol(getRange(symbol).clone(), var, with);
            Set<Symbol> kill = new LinkedHashSet<Symbol>(2);
            kill.add(var);
            kill.add(symbol);
            if (IRTools.containsSymbols(replaced, kill))
                removeRange(symbol);
            else
                setRange(symbol, replaced);
        }
    }

    // Returns the list of subexpressions of e monotonic with var.
    private Expression expandMonoSubexprs(
            Expression e, Symbol var, RangeExpression range) {
        Expression lb, ub, ret = e;
        if (!IRTools.containsSymbol(e, var)) {
            ;
        } else if (isNonDecreasingWith(e, var)) {
            if (range.getLB()instanceof InfExpression)
                lb = new InfExpression(-1);
            else
                lb = Symbolic.simplify(IRTools.
                                       replaceSymbol(e, var,
                                                     range.getLB()));
            if (range.getUB()instanceof InfExpression)
                ub = new InfExpression(1);
            else
                ub = Symbolic.simplify(IRTools.
                                       replaceSymbol(e, var,
                                                     range.getUB()));
            ret = new RangeExpression(lb, ub);
        } else if (isNonIncreasingWith(e, var)) {
            if (range.getUB()instanceof InfExpression)
                lb = new InfExpression(-1);
            else
                lb = Symbolic.simplify(IRTools.
                                       replaceSymbol(e, var,
                                                     range.getUB()));
            if (range.getLB()instanceof InfExpression)
                ub = new InfExpression(1);
            else
                ub = Symbolic.simplify(IRTools.
                                       replaceSymbol(e, var,
                                                     range.getLB()));
            ret = new RangeExpression(lb, ub);
        } else { // go deeper into the expressions tree
            Expression prev = null;
            ret = e;
            while (prev == null || !prev.equals(ret)) {
                prev = ret;
                for (int i = 0; i < ret.getChildren().size(); i++) {
                    Expression child = (Expression)ret.getChildren().get(i);
                    Expression expanded_child =
                            expandMonoSubexprs(child, var, range);
                    if (expanded_child instanceof RangeExpression) {
                        ret.setChild(i, expanded_child);
                        break;
                    }
                }
                ret = expandOnce(ret);
            }
        }
        return ret;
    }

    /**
    * Replace all occurrences of the specified variable in the given expression
    * with the new expression.
    * @param e    the expression being modified
    * @param var  the variable being replaced
    * @param with the new expression being replaced with
    */
    public Expression replaceSymbol(Expression e, Symbol var, Expression with) {
        if (with == null) {
            return (IRTools.containsSymbol(e, var)) ? null : e;
        }
        // Direct simplification
        if (!(with instanceof RangeExpression)) {
            DFIterator<Identifier> iter =
                    new DFIterator<Identifier>(e, Identifier.class);
            while (iter.hasNext()) {
                Identifier id = iter.next();
                if (id.getSymbol() == var) {
                    if (e == id) {
                        return with.clone();
                    } else {
                        id.swapWith(with.clone());
                    }
                }
            }
            return Symbolic.simplify(e);
        }
        Expression ret = e;
        RangeExpression with_range = RangeExpression.toRange(with);
        // High-level expansion for tighter bound.
        if (!with_range.isOmega())
            ret = expandMonoSubexprs(ret, var, with_range);
        // Replace var one-by-one: O(tree-depth*number_of_occurrences)
        Expression prev = null;
        while (prev != ret) {
            prev = ret;
            ret = expand(prev, var, with);
        }
        return ret;
    }

    // In-place expand symbol in the range set.
    protected void expandSymbol(Symbol var) {
        if (!ranges.keySet().contains(var))
            return;
        RangeDomain clone = this.clone();
        for (Symbol symbol : clone.getSymbols())
            setRange(symbol, clone.expandSymbol(getRange(symbol), var));
        return;
    }

    /**
    * Eliminates all occurrences of the specified variables from the expression
    * by replacing them with their value ranges.
    * @param e the expression to be modified.
    * @param vars the set of symbols to be replaced by their ranges.
    */
    public Expression expandSymbols(Expression e, Set<Symbol> vars) {
        if (e == null)
            return RangeExpression.getOmega();
        Expression ret = e.clone();
        Set<Symbol> remain = new LinkedHashSet<Symbol>(vars);
        List<Symbol> order = getReplaceOrder(ret);

        for (Symbol var : order) {
            if (var == null)
                return RangeExpression.getOmega();
            if (remain.contains(var)) {
                ret = expandSymbol(ret, var);
                remain.remove(var);
            }
        }
        for (Symbol var : remain) {
            if (var == null)
                return RangeExpression.getOmega();
            ret = expandSymbol(ret, var);
        }
        return ret;
    }

    /**
    * Eliminates all occurrences of the specified variable from the expression
    * by replacing them with their value ranges.
    * @param e the expression to be modified.
    * @param var the symbol to be replaced by its range.
    */
    public Expression expandSymbol(Expression e, Symbol var) {
        if (e == null)
            return RangeExpression.getOmega();
        Expression ret =
            replaceSymbol(Symbolic.simplify(e), var, getRange(var));
        return (ret == null) ? RangeExpression.getOmega() : ret;
    }

    /**
    * Checks if the given expression is monotonically nondecreasing with the
    * specified variable. This routine is used for high-level range expansion
    * that minimizes the loss of information.
    * e.g.) m*m-m {m=[1:10]} --> [0:90] with the monotonicity information.
    *       m*m-m {m=[1:10]} --> [1:10]*[1:10]-[1:10] = [-9:99]
    *                                   with normal expansion algorithm.
    */
    private boolean isNonDecreasingWith(Expression e, Symbol var) {
        // just return false since e doesn't need to be replaced by var.
        if (!IRTools.containsSymbol(e, var))
            return false;
        Expression index = new Identifier(var);
        Expression next =
            IRTools.replaceSymbol(e, var, Symbolic.increment(index));
        Expression diff = Symbolic.subtract(next, e);
        // TODO: disable & test
        if (diff.toString().length() > e.toString().length())
            return false;       // Give up too complicated differences.
        Relation rel = compareExpressions(diff, 0);
        return rel.isGE();
    }

    /**
    * Checks if the given expression is monotonically nonincreasing with the
    * specified variable.
    */
    private boolean isNonIncreasingWith(Expression e, Symbol var) {
        // just return false since e doesn't need to be replaced by var.
        if (!IRTools.containsSymbol(e, var))
            return false;
        Expression index = new Identifier(var);
        Expression next =
            IRTools.replaceSymbol(e, var, Symbolic.increment(index));
        Expression diff = Symbolic.subtract(next, e);
        // TODO; disable & test
        if (diff.toString().length() > e.toString().length())
            return false;       // Give up too complicated differences.
        Relation rel = compareExpressions(diff, 0);
        return rel.isLE();
    }

    // Compute and return a replacement order after building Range Dependence
    // Graph.
    private List<Symbol> getReplaceOrder(Expression e) {
        List<Symbol> ret = new LinkedList<Symbol>();
        DFAGraph rdg = new DFAGraph();
        Set<Symbol> keyset = getDependentSymbols(e);
        DFANode root = new DFANode("scc-obj", e);
        if (keyset.size() == 0)
            return ret;
        for (Symbol symbol : keyset)
            rdg.addEdge(root, new DFANode("scc-obj", symbol));
        for (Symbol key : ranges.keySet()) {
            for (Symbol symbol : getDependentSymbols(ranges.get(key))) {
                DFANode from = rdg.getNode("scc-obj", key);
                DFANode to = rdg.getNode("scc-obj", symbol);
                if (from == null)
                    from = new DFANode("scc-obj", key);
                if (to == null)
                    to = new DFANode("scc-obj", symbol);
                rdg.addEdge(from, to);
            }
        }
        List scc = rdg.getSCC(root);
        for (Object o : scc) {
            List tree = (List) o;
            if (tree.size() == 1) {
                Object data = ((DFANode) tree.get(0)).getData("scc-obj");
                if (data instanceof Symbol)
                    ret.add(0, (Symbol)data);
            } else { // Heurisitic method that repeats cycles twice.
                List<Symbol> sub_list = new LinkedList<Symbol>();
                for (Object key:tree) {
                    Object data = ((DFANode) key).getData("scc-obj");
                    if (data instanceof Symbol)
                        sub_list.add(0, (Symbol) data);
                }
                scheduleCycle(sub_list, ret);
            }
        }
        return ret;
    }

    // Basic heuristics repeats the cycle twice.
    // Don't follow heuristics for this case (frequent cases in a loop).
    // 1. two symbols have a cycle
    // 2. symbol#1 has a +INF/-INF bound.
    // 3. symbol#2 has a closed bound with one being literals
    // 4. then replace order does not continue on symbol#2->symbol#1
    private void scheduleCycle(List<Symbol> cycle, List<Symbol> ret) {
        if (cycle.size() == 2) {
            RangeExpression range0 =
                    RangeExpression.toRange(getRange(cycle.get(0)));
            RangeExpression range1 =
                    RangeExpression.toRange(getRange(cycle.get(1)));
            if ((range0.getLB()instanceof InfExpression
                 || range0.getUB()instanceof InfExpression)
                && (range1.getLB()instanceof IntegerLiteral
                    || range1.getUB()instanceof IntegerLiteral)) {
                ret.addAll(cycle);
                return;
            } else if ((range1.getLB()instanceof InfExpression ||
                        range1.getUB()instanceof InfExpression) &&
                       (range0.getLB()instanceof IntegerLiteral ||
                        range0.getUB()instanceof IntegerLiteral)) {
                ret.add(cycle.get(0));
                return;
            }
        }
        ret.addAll(cycle);
        ret.addAll(cycle);
    }

    // Split the expression into set of key expressions that can be used as 
    // keys in the range domain
    private Set<Symbol> getDependentSymbols(Expression e) {
        Set<Symbol> ret = new LinkedHashSet<Symbol>();
        if (e instanceof Identifier) {
            Symbol var = ((Identifier)e).getSymbol();
            if (ranges.containsKey(var))
                ret.add(var);
        } else if (e instanceof BinaryExpression) {
            BinaryOperator op = ((BinaryExpression)e).getOperator();
            if (op == BinaryOperator.ADD || op == BinaryOperator.DIVIDE ||
                op == BinaryOperator.MULTIPLY
                || op == BinaryOperator.SUBTRACT
                || op == BinaryOperator.MODULUS) {
                ret.addAll(getDependentSymbols(((BinaryExpression)e).getLHS()));
                ret.addAll(getDependentSymbols(((BinaryExpression)e).getRHS()));
            }
        } else if (e instanceof RangeExpression
                 || e instanceof MinMaxExpression) {
            for (Traversable o : e.getChildren())
                ret.addAll(getDependentSymbols((Expression)o));
        }
        return ret;
    }

    // Expand the given expression after replacing the first occurrence of "var"
    // with "with". Here expand means RangeExpression is pulled up as much as
    // possible (usually up to the root of the expression tree).
    private Expression expand(Expression e, Symbol var, Expression with) {
        Identifier marker = null;
        DFIterator<Identifier> iter =
                new DFIterator<Identifier>(e, Identifier.class);
        while (iter.hasNext()) {
            Identifier id = iter.next();
            if (id.getSymbol() == var) {
                marker = id;
                break;
            }
        }
        // No more expansion - guarantees termination
        if (marker == null)
            return e;
        // Return a copy of the expression directly if the expressions is an
        // identifier
        if (marker == e)
            return with.clone();
        // Replace first
        Expression parent = (Expression)marker.getParent();
        marker.swapWith(with.clone());
        // Expand the replaced range up to the root of the expression tree
        while (parent != e && parent != null) {
            Expression before = parent;
            parent = (Expression)parent.getParent();
            Expression expanded = expandOnce(before);
            before.swapWith(expanded);
        }
        // Final expansion at the top-level
        Expression ret = expandOnce(parent);
        return ret;
    }

    // Single expansion for the given expression
    private Expression expandOnce(Expression e) {
        Expression ret = null;
        if (e instanceof RangeExpression) {
            RangeExpression re = (RangeExpression)e;
            ret = expandRange(re.getLB(), re.getUB());
        } else if (e instanceof BinaryExpression) {
            BinaryExpression be = (BinaryExpression)e;
            Expression l = be.getLHS(), r = be.getRHS();
            if (be.getOperator() == BinaryOperator.ADD) {
                ret = expandADD(l, r);
            } else if (be.getOperator() == BinaryOperator.MULTIPLY) {
                ret = expandMUL(l, r);
            } else if (be.getOperator() == BinaryOperator.DIVIDE) {
                ret = expandDIV(l, r);
            } else if (be.getOperator() == BinaryOperator.MODULUS) {
                ret = expandMOD(l, r);
            } else if (!(l instanceof RangeExpression) &&
                       !(r instanceof RangeExpression)) {
                ret = be.clone();
            }
        } else if (e instanceof MinMaxExpression) {
            ret = expandMinMax((MinMaxExpression)e);
        }
        if (ret == null) {
            ret = RangeExpression.getOmega();
        }
        return ret;
    }

    // [e1:e2] => [e1.lb:e2.ub]
    private Expression expandRange(Expression e1, Expression e2) {
        Expression lb = ((e1 instanceof RangeExpression) ?
                            ((RangeExpression)e1).getLB().clone() :
                            e1.clone());
        Expression ub = ((e2 instanceof RangeExpression) ?
                            ((RangeExpression)e2).getUB().clone() :
                            e2.clone());
        return new RangeExpression(lb, ub);
    }

    // [e1:e2] => [e1.lb+e2.lb:e1.ub+e2.ub]
    private Expression expandADD(Expression e1, Expression e2) {
        RangeExpression re1 = RangeExpression.toRange(e1);
        RangeExpression re2 = RangeExpression.toRange(e2);
        Expression lb1 = re1.getLB(), ub1 = re1.getUB();
        Expression lb2 = re2.getLB(), ub2 = re2.getUB();
        Expression lb, ub;
        if (lb1 instanceof InfExpression) {
            lb = lb1;
        } else if (lb2 instanceof InfExpression) {
            lb = lb2;
        } else {
            lb = Symbolic.add(lb1, lb2);
        }
        if (ub1 instanceof InfExpression) {
            ub = ub1;
        } else if (ub2 instanceof InfExpression) {
            ub = ub2;
        } else {
            ub = Symbolic.add(ub1, ub2);
        }
        lb.setParent(null);
        ub.setParent(null);
        return new RangeExpression(lb, ub);
    }

    // [a:b]*c => [a*c:b*c]   if c>=0,
    //            [b*c:a*c]   if c<=0,
    //            [-inf:inf]  otherwise.
    private Expression expandMUL(Expression e1, Expression e2) {
        RangeExpression re;
        Expression e;
        // Identify RangeExpression
        if (e1 instanceof RangeExpression) {
            if (e2 instanceof RangeExpression) {
                return RangeExpression.getOmega();
            } else {
                re = (RangeExpression)e1;
                e = e2;
            }
        } else if (e2 instanceof RangeExpression) {
            re = (RangeExpression)e2;
            e = e1;
        } else {
            return Symbolic.multiply(e1, e2);
        }

        Relation e_sign = compareExpressions(e, 0);
        boolean e_positive = e_sign.isGE(), e_negative = e_sign.isLE();

        // Give up unknown result; e being InfExpression can progress further
        // if a>=0 in [a:b].
        if (!e_positive && !e_negative) {
            return RangeExpression.getOmega();
        }

        if (e instanceof InfExpression) {
            if (re.getLB().equals(zero)) {
                if (e_positive)
                    return new RangeExpression(new IntegerLiteral(0),
                                               new InfExpression(1));
                else
                    return new RangeExpression(new InfExpression(-1),
                                               new IntegerLiteral(0));
            } else if (re.getUB().equals(zero)) {
                if (e_positive)
                    return new RangeExpression(new InfExpression(-1),
                                               new IntegerLiteral(0));
                else
                    return new RangeExpression(new IntegerLiteral(0),
                                               new InfExpression(1));
            } else
                return RangeExpression.getOmega();
        }
        // Adjust lb ub position w.r.t. the sign of the multiplier
        Expression lb = re.getLB(), ub = re.getUB();
        if (e_negative) {
            lb = re.getUB();
            ub = re.getLB();
        }
        // Lower bound
        if (lb instanceof InfExpression) {
            lb = new InfExpression(-1);
        } else if (lb instanceof MinMaxExpression) {
            MinMaxExpression mme = (MinMaxExpression)lb;
            lb = mme.clone();
            List<Traversable> children = lb.getChildren();
            for (int i = 0; i < children.size(); ++i) {
                lb.setChild(i,
                            Symbolic.multiply(e, (Expression)children.get(i)));
            }
            if (e_negative) {
                mme.setMin(!mme.isMin());
            }
        } else {
            lb = Symbolic.multiply(e, lb);
        }
        // Upper bound
        if (ub instanceof InfExpression) {
            ub = new InfExpression(1);
        } else if (ub instanceof MinMaxExpression) {
            MinMaxExpression mme = (MinMaxExpression)ub;
            ub = mme.clone();
            List<Traversable> children = ub.getChildren();
            for (int i = 0; i < children.size(); ++i) {
                ub.setChild(i,
                            Symbolic.multiply(e, (Expression)children.get(i)));
            }
            if (e_negative) {
                mme.setMin(!mme.isMin());
            }
        } else {
            ub = Symbolic.multiply(e, ub);
        }
        return new RangeExpression(lb, ub);
    }

    private Expression expandDIV(Expression e1, Expression e2) {
        // [a:b]/c => [a/c:b/c]   if c>0,
        //            [b/c:a/c]   if c<0,
        //            [-INF:+INF] otherwise
        if (e1 instanceof RangeExpression) {
            if (e2 instanceof RangeExpression)
                return RangeExpression.getOmega();
            int e2_sign = signOf(e2);
            if (e2_sign == 999 || e2_sign == 0)
                return RangeExpression.getOmega();
            RangeExpression re1 = (RangeExpression) e1;
            Expression lb = re1.getLB(), ub = re1.getUB();
            if (e2_sign < 0) {
                lb = re1.getUB();
                ub = re1.getLB();
            }
            // Lower bound
            if (lb instanceof InfExpression) {
                lb = new InfExpression(-1);
            } else if (lb instanceof MinMaxExpression) {
                MinMaxExpression mme = (MinMaxExpression)lb;
                lb = mme.clone();
                List<Traversable> children = lb.getChildren();
                for (int i = 0; i < children.size(); ++i) {
                    lb.setChild(i, Symbolic.divide(
                            (Expression)children.get(i), e2));
                }
                if (e2_sign < 0) {
                    mme.setMin(!mme.isMin());
                }
            } else {
                lb = Symbolic.divide(lb, e2);
            }
            // Upper bound
            if (ub instanceof InfExpression) {
                ub = new InfExpression(1);
            } else if (ub instanceof MinMaxExpression) {
                MinMaxExpression mme = (MinMaxExpression)ub;
                ub = mme.clone();
                List<Traversable> children = ub.getChildren();
                for (int i = 0; i < children.size(); ++i) {
                    ub.setChild(i, Symbolic.divide(
                            (Expression)children.get(i), e2));
                }
                if (e2_sign < 0) {
                    mme.setMin(!mme.isMin());
                }
            } else {
                ub = Symbolic.divide(ub, e2);
            }
            return new RangeExpression(lb, ub);
        } else if (e2 instanceof RangeExpression) {
            // c/[a:b] => [c/a:c/b]   if c<0 and (a>0||b<0),
            //            [c/b:c/a]   if c>0 and (a>0||b<0),
            //            [-INF:+INF] otherwise
            int e1_sign = signOf(e1);
            int e2_sign = signOf(e2);
            if (e2_sign == 999 || e2_sign == 0 || e1_sign == 999)
                return RangeExpression.getOmega();
            if (e1_sign == 0)
                return new IntegerLiteral(0);
            RangeExpression re2 = (RangeExpression)e2;
            Expression lb = re2.getLB(), ub = re2.getUB();
            if (e1_sign > 0) {
                lb = re2.getUB();
                ub = re2.getLB();
            }
            // Lower bound
            if (lb instanceof InfExpression) {
                lb = new IntegerLiteral(0);
            } else if (lb instanceof MinMaxExpression) {
                MinMaxExpression mme = (MinMaxExpression)lb;
                lb = mme.clone();
                List<Traversable> children = lb.getChildren();
                for (int i = 0; i < children.size(); ++i) {
                    lb.setChild(i, Symbolic.divide(
                            e1, (Expression)children.get(i)));
                }
                if (e1_sign > 0) {
                    mme.setMin(!mme.isMin());
                }
            } else {
                lb = Symbolic.divide(e1, lb);
            }
            // Upper bound
            if (ub instanceof InfExpression) {
                ub = new IntegerLiteral(0);
            } else if (ub instanceof MinMaxExpression) {
                MinMaxExpression mme = (MinMaxExpression)ub;
                ub = mme.clone();
                List<Traversable> children = ub.getChildren();
                for (int i = 0; i < children.size(); ++i) {
                    ub.setChild(i, Symbolic.divide(
                            e1, (Expression)children.get(i)));
                }
                if (e1_sign > 0) {
                    mme.setMin(!mme.isMin());
                }
            } else {
                ub = Symbolic.divide(e1, ub);
            }
            return new RangeExpression(lb, ub);
        } else {
            return Symbolic.divide(e1, e2);
        }
    }

    // expandMOD reimplemented
    private Expression expandMOD(Expression dividend, Expression divider) {
        Expression abs_of_divider = abs(divider);
        if (abs_of_divider == null || !isGT(abs_of_divider, zero))
            return RangeExpression.getOmega();  // potential division by zero
        RangeExpression divider_re = RangeExpression.toRange(abs_of_divider);
        Expression positive_bound = Symbolic.decrement(divider_re.getUB());
        Expression negative_bound = Symbolic.subtract(zero, positive_bound); 
        RangeExpression ret =
            new RangeExpression(negative_bound, positive_bound);
        Relation dividend_rel = compareExpressions(dividend, 0);
        if (dividend_rel.isGE())
            ret.setLB(new IntegerLiteral(0));
        else if (dividend_rel.isLE())
            ret.setUB(new IntegerLiteral(0));
        return ret;
    }

    // Expansion for mod expressions
    private Expression expandMOD2(Expression l, Expression r) {
        RangeExpression re = (RangeExpression)
            ((l instanceof RangeExpression) ? l : r);
        Expression other = (l instanceof RangeExpression) ? r : l;

        Relation rel = compareExpressions(other, 0);
        Relation rellb = compareExpressions(re.getLB(), 0);
        Relation relub = compareExpressions(re.getUB(), 0);
        boolean positive_dividend = false, negative_dividend = false;
        Expression abs = null;
        Expression lb = new InfExpression(-1), ub = new InfExpression(1);

        if (l instanceof RangeExpression) {
            if (rel.isGT())
                abs = other;
            else if (rel.isLT())
                abs = Symbolic.multiply(new IntegerLiteral(-1), other);
            positive_dividend = rellb.isGE();
            negative_dividend = relub.isLE();
        } else {
            if (rellb.isGT())
                abs = re.getUB().clone();
            else if (relub.isLT())
                abs = Symbolic.multiply(new IntegerLiteral(-1), re.getLB());
            positive_dividend = rel.isGE();
            negative_dividend = rel.isLE();
        }

        // No division is possible
        if (abs == null)
            return RangeExpression.getOmega();

        // Range is defined by the divisor's maximum absolute value and
        // the sign of dividend can further narrow the range
        lb = Symbolic.multiply(new IntegerLiteral(-1), abs);
        lb = Symbolic.add(lb, new IntegerLiteral(1));
        ub = Symbolic.subtract(abs, new IntegerLiteral(1));

        if (positive_dividend)
            lb = new IntegerLiteral(0);

        else if (negative_dividend)
            ub = new IntegerLiteral(0);

        return new RangeExpression(lb, ub);
    }

    // Expand min/max expression
    private Expression expandMinMax(MinMaxExpression e) {
        MinMaxExpression lb = new MinMaxExpression(e.isMin());
        MinMaxExpression ub = new MinMaxExpression(e.isMin());

        for (Traversable o : e.getChildren()) {
            Expression child = (Expression)o;
            for (int i = 0; i < 2; ++i) {
                List<Expression> temp = new ArrayList<Expression>();
                MinMaxExpression bound = (i == 0) ? lb : ub;
                if (child instanceof RangeExpression) {
                    temp.add((Expression)child.getChildren().get(i));
                } else {
                    temp.add(child);
                }
                for (Expression expr : temp) {
                    if (expr instanceof MinMaxExpression &&
                        ((MinMaxExpression)expr).isMin() == e.isMin()) {
                        for (Object oo : expr.getChildren()) {
                            bound.add((Expression)oo);
                        }
                    } else {
                        bound.add(expr);
                    }
                }
            }
        }
        Expression ret = new RangeExpression(lb, ub);
        return Symbolic.simplify(ret);
    }

  /*====================================================================
    Methods for abstract operations; intersect, union, widen, and narrow
    ====================================================================*/

    /**
    * Intersects two sets of value ranges using current range domain.
    * @param other the range domain intersected with
    */
    public void intersectRanges(RangeDomain other) {
        RangeDomain before = null;
        if (debug >= 2) {
            before = this.clone();
        }
        // Copy explicit intersections first
        for (Symbol var : other.ranges.keySet()) {
            if (getRange(var) == null) {
                setRange(var, other.getRange(var).clone());
            }
        }
        for (Symbol var : new LinkedHashSet<Symbol>(ranges.keySet())) {
            Expression result = intersectRanges(getRange(var), this,
                                                other.getRange(var), this);
            // Removing empty ranges trigger infeasible paths
            if (isEmpty(result, this)) {
                ranges.clear();
                break;
            }
            if (isOmega(result)) {
                removeRange(var);
            } else {
                setRange(var, result);
            }
        }
        PrintTools.printlnStatus(2, tag, before, "(^)", other, "=", this);
    }

    /**
    * Merges two sets of value ranges using current range domain (union
    * operation).
    * @param other the range domain merged with
    */
    public void unionRanges(RangeDomain other) {
        RangeDomain before = null;
        if (debug >= 2) {
            before = this.clone();
        }
        //String dmsg = tag + this + " (v) " + other;
        for (Symbol var : new LinkedHashSet<Symbol>(ranges.keySet())) {
            Expression result = unionRanges(getRange(var), this,
                                            other.getRange(var), other);
            if (isOmega(result)) {
                removeRange(var);
            } else {
                setRange(var, result);
            }
        }
        PrintTools.printlnStatus(2, tag, before, "(v)", other, "=", this);
    }

    /**
    * Widens all value ranges of "other" range domain with this range domain.
    * @param other value ranges being widened
    */
    public void widenRanges(RangeDomain other) {
        widenAffectedRanges(other,
                new LinkedHashSet<Symbol>(other.ranges.keySet()));
    }

    /**
    * Widens subset of value ranges in "other" that contains the specified
    * symbols either in keys or in value ranges.
    * @param other the range domain containing widening operands
    * @param vars  set of symbols that trigger widening
    */
    public void widenAffectedRanges(RangeDomain other, Set<Symbol> vars) {
        RangeDomain before = null;
        if (debug >= 2) {
            before = this.clone();
        }
        //String dmsg = tag + other + " (w) " + this;
        Set<Symbol> affected = new LinkedHashSet<Symbol>();
        for (Symbol var_range : other.ranges.keySet()) {
            for (Symbol var_in : vars) {
                if (IRTools.containsSymbol(getRange(var_range), var_in))
                    affected.add(var_range);
            }
        }
        affected.addAll(vars);
        for (Symbol var : affected) {
            Expression result =
                widenRange(other.getRange(var), getRange(var), this);
            if (isOmega(result)) {
                removeRange(var);
            } else {
                setRange(var, result);
            }
        }
        PrintTools.printlnStatus(2, tag, other, "(w)", before, "=", this);
    }

    /**
    * Narrows all value ranges of "other" range domain with this range domain.
    * @param other value ranges being narrowed.
    */
    public void narrowRanges(RangeDomain other) {
        RangeDomain before = null;
        if (debug >= 2) {
            before = this.clone();
        }
        for (Symbol var : other.ranges.keySet()) {
            Expression result =
                narrowRange(other.getRange(var), getRange(var), this);
            if (isOmega(result)) {
                removeRange(var);
            } else {
                setRange(var, result);
            }
        }
        PrintTools.printlnStatus(2, tag, other, "(n)", before, "=", this);
    }

    /**
    * Computes the intersection of the two expressions with the given range
    * domains.
    * @param e1  first expression
    * @param rd1 first range domain
    * @param e2  second expression
    * @param rd2 second range domain
    * @return    intersection of the two expressions
    */
    public static Expression intersectRanges
        (Expression e1, RangeDomain rd1, Expression e2, RangeDomain rd2) {
        // [lb:ub] = [a:b] ^ [c:d],    lb = a        if a>=c
        //                                = c        if a<c
        //                                = a        otherwise (ACCURACY=0)
        //                                = heuristics         (ACCURACY=1)
        //                                = max(a,c)           (ACCURACY=2)
        //                             ub = b        if b<=d
        //                                = d        if b>d
        //                                = b        otherwise (ACCURACY=0)
        //                                = heurisitcs         (ACCURACY=1) 
        //                                = min(b,d)           (ACCURACY=2)
        //
        // Check if e1/e2 is unknown range, empty range, or they are equal.
        if (isOmega(e1))
            return (isOmega(e2)) ? null : e2.clone();
        else if (isOmega(e2))
            return e1.clone();
        else if (isEmpty(e1))
            return e1.clone();
        else if (isEmpty(e2))
            return e2.clone();
        else if (e1.compareTo(e2) == 0)
            return e1.clone();

        // Converts e1 & e2 to range expressions. re1 and re2 contain cloned
        // copies of e1 and e2 in their lb and ub.
        RangeExpression re1 = RangeExpression.toRange(e1);
        RangeExpression re2 = RangeExpression.toRange(e2);
        Expression lb1 = re1.getLB(), lb2 = re2.getLB(), lb = null;
        Expression ub1 = re1.getUB(), ub2 = re2.getUB(), ub = null;
        Relation lbrel = compare(lb1, rd1, lb2, rd2);
        Relation ubrel = compare(ub1, rd1, ub2, rd2);
        Expression ret = null;

        // Compare lower bounds and take MAX
        if (lbrel.isGE()) {
            lb = lb1;
        } else if (lbrel.isLT()) {
            lb = lb2;
        } else if ((lbrel = compare(lb2, rd2, lb1, rd1)).isGE()) {
            lb = lb2;
        } else if (lbrel.isLT()) {
            lb = lb1;
        } else if (ACCURACY < 1) {
            lb = lb1;
        } else if (ACCURACY < 2) {
            Relation rel = compare(ub1, rd1, lb2, rd2);
            if (rel.isLT()) {      // lb1 <= ub1 < lb2 <= ub2
                ret = new RangeExpression(one.clone(), neg_one.clone());
                // empty range
            } else if (rel.isEQ()) { // lb1 <= ub1 == lb2 <= ub2
                ret = lb2;
            } else {
                rel = compare(ub2, rd2, lb1, rd1);
                if (rel.isLT()) { // lb2 <= ub2 < lb1 <= ub1
                    ret = new RangeExpression(one.clone(), neg_one.clone());
                } else if (rel.isEQ()) { // lb2 <= ub2 == lb1 <= ub1
                    ret = lb1;
                } else {
                    lb = lb1;
                }
            }
        } else {
            lb1.setParent(null);
            lb2.setParent(null);
            lb = Symbolic.simplify(new MinMaxExpression(false, lb1, lb2));
        }

        if (ret != null) {
            ret.setParent(null);
            return ret;
        }

        // Compare upper bounds and take MIN
        if (ubrel.isLE()) {
            ub = ub1;
        } else if (ubrel.isGT()) {
            ub = ub2;
        } else if ((ubrel = compare(ub2, rd2, ub1, rd1)).isLE()) {
            ub = ub2;
        } else if (ubrel.isGT()) {
            ub = ub1;
        } else if (ACCURACY < 2) {
            ub = ub1;
        } else {
            ub1.setParent(null);
            ub2.setParent(null);
            ub = Symbolic.simplify(new MinMaxExpression(true, ub1, ub2));
        }
/* Enable if ncessary
        // Detect MAX(a,b):MIN(a,b); just return the first expression.
        if (lb instanceof MinMaxExpression
            && ub instanceof MinMaxExpression
            && !((MinMaxExpression)lb).isMin()
            && ((MinMaxExpression)ub).isMin()
            && compareChildren(lb, ub) == 0) {
            return re1.getLB();
        }
*/
        lb.setParent(null);
        ub.setParent(null);
        RangeExpression range = new RangeExpression(lb, ub);
        ret = range.toExpression();

/* Enable if necessary
        // Heuristics to avoid more complicated tight bounds.
        if (!isEmpty(ret, rd1) && !ret.equals(e1) &&
            ret instanceof RangeExpression &&
            re1.isBounded() && range.isBounded()) {
            Expression lb_diff = Symbolic.subtract(lb1, lb);
            Expression ub_diff = Symbolic.subtract(ub1, ub);
            if (!(lb_diff instanceof IntegerLiteral &&
                  ub_diff instanceof IntegerLiteral))
                ret = e1;
        }
*/
        return ret;
    }

    /**
    * Computes the union of the two expressions with the given range domains.
    * @param e1  first expression
    * @param rd1 first range domain
    * @param e2  second expression
    * @param rd2 second range domain
    * @return    union of the two expressions
    */
    public static Expression unionRanges
        (Expression e1, RangeDomain rd1, Expression e2, RangeDomain rd2) {
        // [lb:ub] = [a:b] U [c:d],    lb = a        if a<=c
        //                                = c        if a>c
        //                                = -INF     otherwise (ACCURACY=0)
        //                                = heuristics         (ACCURACY=1)
        //                                = min(a,c)           (ACCURACY=2) 
        //                             ub = b        if b>=d
        //                                = d        if b<d
        //                                = +INF     otherwise (ACCURACY=0)
        //                                = heuristics         (ACCURACY=1)
        //                                = max(b,d)           (ACCURACY=2)
        //
        // Check if either e1/e2 is omega range, empty range, or e1==e2.
        if (isOmega(e1) || isOmega(e2))
            return null;
        else if (isEmpty(e1))
            return e2.clone();
        else if (isEmpty(e2))
            return e1.clone();
        else if (e1.compareTo(e2) == 0)
            return e1.clone();

        // Converts e1 & e2 to range expressions.
        // re1 and re2 contain cloned copies of e1 and e2 in their lb and ub.
        RangeExpression re1 = RangeExpression.toRange(e1);
        RangeExpression re2 = RangeExpression.toRange(e2);
        Expression lb1 = re1.getLB(), lb2 = re2.getLB(), lb = null;
        Expression ub1 = re1.getUB(), ub2 = re2.getUB(), ub = null;
        Relation lbrel = compare(lb1, rd1, lb2, rd2);
        Relation ubrel = compare(ub1, rd1, ub2, rd2);

        // Compare lower bounds and take MIN
        if (lbrel.isLE()) {
            lb = lb1;
        } else if (lbrel.isGE()) {
            lb = lb2;
        } else if (ACCURACY < 1) {
            lb = new InfExpression(-1);
        } else if (ACCURACY < 2) { // heuristics
            // heuristic1: explicit limit value
            // [a,b] v [b,b] = [a,b] x
            // [a,b] v [a,a] = [a,b]
            // [a,a] v [a,b] = [a,b]
            // [b,b] v [a,b] = [a,b] x
            if ((compare(ub1, rd1, ub2, rd2)).isEQ() &&
                (compare(lb2, rd2, ub2, rd2)).isEQ()) {
                lb = lb1;
            } else if ((compare(lb1, rd1, ub1, rd1)).isEQ() &&
                     (compare(ub1, rd1, ub2, rd2)).isEQ()) {
                lb = lb2;
            } else if ((compare(ub1, rd1, lb2, rd2)).isLE()) {
                // heuristic2: non-overlapping ranges
                // ( [a,b] v [b+k, d] = [a, d] )
                if ((compare(lb1, rd1, ub2, rd2)).isLE())
                    lb = lb1;
                else
                    lb = new InfExpression(-1);
            } else { // heuristic3: bound to zero
                Relation sign1 = compare(lb1, rd1, zero, rd2);
                Relation sign2 = compare(lb2, rd2, zero, rd1);
                if (sign1.isGE() && sign2.isGE())
                    lb = zero.clone();
                else
                    lb = new InfExpression(-1);
            }
        } else {
            lb1.setParent(null);
            lb2.setParent(null);
            lb = Symbolic.simplify(new MinMaxExpression(true, lb1, lb2));
        }

        // Compare upper bounds and take MAX
        if (ubrel.isGE()) {
            ub = ub1;
        } else if (ubrel.isLE()) {
            ub = ub2;
        } else if (ACCURACY < 1) {
            ub = new InfExpression(1);
        } else if (ACCURACY < 2) {
            // heuristic1: explicit limit value
            // [a,b] v [b,b] = [a,b]
            // [a,b] v [a,a] = [a,b] x
            // [a,a] v [a,b] = [a,b] x
            // [b,b] v [a,b] = [a,b]
            if ((compare(lb1, rd1, lb2, rd2)).isEQ() &&
                (compare(lb2, rd2, ub2, rd2)).isEQ()) {
                ub = ub1;
            } else if ((compare(lb1, rd1, ub1, rd1)).isEQ() &&
                     (compare(lb1, rd1, lb2, rd2)).isEQ()) {
                ub = ub2;
            } else if ((compare(ub1, rd1, lb2, rd2)).isLE()) {
                if ((compare(lb1, rd1, ub2, rd2)).isLE())
                    ub = ub2;
                else
                    ub = new InfExpression(1);
            } else {
                Relation sign1 = compare(ub1, rd1, zero, rd2);
                Relation sign2 = compare(ub2, rd2, zero, rd1);
                if (sign1.isLE() && sign2.isLE())
                    ub = zero.clone();
                else
                    ub = new InfExpression(1);
            }
        } else {
            ub1.setParent(null);
            ub2.setParent(null);
            ub = Symbolic.simplify(new MinMaxExpression(false, ub1, ub2));
        }

        lb.setParent(null);
        ub.setParent(null);
        if (lb.compareTo(ub) == 0)
            return lb;
        else
            return new RangeExpression(lb, ub);
    }

    // Compute widening operation
    private static Expression widenRange
        (Expression e, Expression widen, RangeDomain rd) {
        // [lb:ub] = [a:b] W [c:d],    lb = a        if a=c
        //                                = -INF     otherwise
        //                             ub = b        if b=d
        //                                = +INF     otherwise
        //
        // Check if e1/e2 is empty or omega.
        if (isOmega(e) || isOmega(widen))
            return null;
        else if (isEmpty(e))
            return widen.clone();
        else if (isEmpty(widen))
            return e.clone();

        // Convert the two expressions to range expressions.
        RangeExpression re = RangeExpression.toRange(e);
        RangeExpression rwiden = RangeExpression.toRange(widen);

        // Compare lower bounds
        Relation rel = rd.compare(re.getLB(), rwiden.getLB());
        if (!rel.isEQ())
            re.setLB(new InfExpression(-1));

        // Compare upper bounds
        rel = rd.compare(re.getUB(), rwiden.getUB());
        if (!rel.isEQ())
            re.setUB(new InfExpression(1));

        if (re.getLB().compareTo(re.getUB()) == 0) {
            Expression ret = re.getLB();
            ret.setParent(null);
            return ret;
        } else {
            return re;
        }
    }

    // Compute narrowing operation
    private static Expression narrowRange
        (Expression e, Expression narrow, RangeDomain rd) {
        // [lb:ub] = [a:b] N [c:d],    lb = a        if a != -INF
        //                                = c        otherwise
        //                             ub = b        if b != +INF
        //                                = d        otherwise
        //
        // Check if operation is singular
        if (isOmega(narrow))
            return (e == null) ? null : e.clone();
        else if (isOmega(e) || !(narrow instanceof RangeExpression))
            return narrow.clone();

        // Convert the two expressions to range expressions.
        RangeExpression re = RangeExpression.toRange(e);
        RangeExpression rnarrow = RangeExpression.toRange(narrow);

        if (re.getLB()instanceof InfExpression) {
            Expression lb = rnarrow.getLB();
            lb.setParent(null);
            re.setLB(lb);
        }

        if (re.getUB()instanceof InfExpression) {
            Expression ub = rnarrow.getUB();
            ub.setParent(null);
            re.setUB(ub);
        }

        if (re.getLB().compareTo(re.getUB()) == 0) {
            Expression ret = re.getLB();
            ret.setParent(null);
            return ret;
        } else {
            return re;
        }
    }


  /*====================================================================
    Miscellaneous helper methods
    ====================================================================*/

    // Converts min/max expression to conditional expression.
    private static void removeMinMax(Expression e) {
        if (e == null)
            return;
        FlatIterator iter = new FlatIterator(e);
        while (iter.hasNext())
            removeMinMax((Expression)iter.next());
        if (e instanceof MinMaxExpression)
            e.swapWith(((MinMaxExpression)e).toConditionalExpression());
    }

    // Test if an expression is omega
    private static boolean isOmega(Expression e) {
        if (e == null || e instanceof InfExpression)
            return true;
        else if (e instanceof RangeExpression)
            return ((RangeExpression)e).isOmega();
        else
            return false;
    }

    // Test if an expression is numerically empty
    private static boolean isEmpty(Expression e) {
        return (e instanceof RangeExpression
                && ((RangeExpression)e).isEmpty());
    }

    // Test if an expression is symbolically empty
    private static boolean isEmpty(Expression e, RangeDomain rd) {
        if (isEmpty(e))
            return true;
        if (!(e instanceof RangeExpression))
            return false;
        RangeExpression re = (RangeExpression) e;
        Relation rel = rd.compare(re.getLB(), re.getUB());
        return (rel.isGT());
    }

    public boolean isEmptyRange(Expression e) {
        if (isEmpty(e))
            return true;
        if (!(e instanceof RangeExpression))
            return false;
        RangeExpression re = (RangeExpression) e;
        Relation rel = compare(re.getLB(), re.getUB());
        return (rel.isGT());
    }

    public boolean encloses(Expression e1, Expression e2) {
        RangeExpression re1 = RangeExpression.toRange(e1);
        RangeExpression re2 = RangeExpression.toRange(e2);
        return (compare(re1.getLB(), re2.getLB()).isLE() &&
                compare(re1.getUB(), re2.getUB()).isGE());
    }

    // Compare equality of the children of the two expressions.
    private static int compareChildren(Expression e1, Expression e2) {
        List child1 = e1.getChildren(), child2 = e2.getChildren();

        if (child1.size() != child2.size())
            return (child1.size() > child2.size())? 1 : -1;

        for (int i = 0, ret = 0; i < child1.size(); ++i) {
            ret = ((Expression)child1.get(i)).compareTo(
                    (Expression)child2.get(i));
            if (ret != 0)
                return ret;
        }
        return 0;
    }

  /*====================================================================
    Methods implementing Domain interface
    ====================================================================*/

    public Domain union (Domain other) {
        if (other instanceof RangeDomain) {
            RangeDomain ret = clone();
            ret.unionRanges((RangeDomain) other);
            return ret;
        } else
            return NullDomain.getNull();
    }

    public Domain merge(Domain other) {
        return union (other);
    }

    public Domain intersect(Domain other) {
        RangeDomain ret = clone();
        if (other instanceof RangeDomain)
            ret.intersectRanges((RangeDomain) other);
        return ret;
    }

    public Domain diffStrong(Domain other) {
        return NullDomain.getNull();
    }

    public Domain diffWeak(Domain other) {
        return NullDomain.getNull();
    }

    public void kill(Set < Symbol > vars) {
        removeRangeWith(vars);
    }

    public boolean equals(RangeDomain other) {
        return equals((Domain) other);
    }

    /**
    * Pairwise comparison between two RangeDomain objects.
    * @param other  the RangeDomain object being compared to this object.
    * @return true if they are equal, false otherwise.
    */
    public boolean equals(Domain other) {
        if (other == null || !(other instanceof RangeDomain))
            return false;
        RangeDomain other_rd = (RangeDomain) other;
        if (ranges.size() != other_rd.ranges.size())
            return false;
        for (Symbol var : ranges.keySet()) {
            Expression range1 = getRange(var), range2 = other_rd.getRange(var);
            if (range2 == null || !range1.equals(range2))
                return false;
        }
        return true;
    }

    /**
    * Kill any var:range mappings containing local variables.
    */
    public void killLocal() {
        killLocalExcept(null);
    }

    /**
    * Kills any ranges containing local variables except for the specified set
    * of variables.
    * @param except the set of variables not considered for kill.
    */
    public void killLocalExcept(Set<Symbol> except) {
        for (Symbol var : new HashSet<Symbol>(ranges.keySet())) {
            Set<Symbol> acc_vars = new HashSet<Symbol>();
            acc_vars.add(var);
            acc_vars.addAll(SymbolTools.getAccessedSymbols(getRange(var)));
            if (except != null)
                acc_vars.removeAll(except);
            for (Symbol acc_var : acc_vars) {
                if (SymbolTools.isLocal(acc_var)) {
                    ranges.remove(var);
                    break;
                }
            }
        }
    }

    /**
    * Kills any ranges containing global variables.
    */
    public void killGlobal() {
        killGlobalAnd(null);
    }

    /**
    * Kills any ranges containing global variables and the specified set of
    * variables.
    * @param and the additional variables to be considered for kill.
    */
    public void killGlobalAnd(Set<Symbol> and) {
        for (Symbol var : new HashSet<Symbol>(ranges.keySet())) {
            Set<Symbol> acc_vars = new HashSet<Symbol>();
            acc_vars.add(var);
            acc_vars.addAll(SymbolTools.getAccessedSymbols(getRange(var)));
            if (and != null)
                acc_vars.addAll(and);
            for (Symbol acc_var : acc_vars) {
                if (SymbolTools.isGlobal(acc_var)) {
                    ranges.remove(var);
                    break;
                }
            }
        }
    }

    /**
    * Kills any ranges containing variables from which Program node is not
    * accessible. Any formal parameters and other symbols for temporary purposes
    * are killed.
    */
    public void killOrphan() {
        for (Symbol var : new HashSet<Symbol>(ranges.keySet())) {
            Set<Symbol> acc_vars = new HashSet<Symbol>();
            acc_vars.add(var);
            acc_vars.addAll(SymbolTools.getAccessedSymbols(getRange(var)));
            for (Symbol acc_var : acc_vars) {
                if (SymbolTools.isOrphan(acc_var)) {
                    ranges.remove(var);
                    break;
                }
            }
        }
    }

    private static List<Object>
            getSignature(RangeDomain rd, Expression e1, Expression e2) {
        List<Object> ret = new ArrayList<Object>(3);
        Map<String, Expression> constraint = new HashMap<String, Expression>();
        for (Symbol var : rd.ranges.keySet()) {
            constraint.put(var.getSymbolName(), rd.ranges.get(var));
        }
        ret.add(constraint);
        ret.add(e1);
        ret.add(e2);
        return ret;
    }
}
