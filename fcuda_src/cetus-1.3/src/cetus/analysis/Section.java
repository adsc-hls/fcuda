package cetus.analysis;

import cetus.hir.*;

import java.util.*;

/**
 * Class Section represents a list of array subscripts that expresses a subset
 * of the whole array elements. Each element in the list should be an object
 * of {@link Section.ELEMENT}.
 *
 * @see Section.ELEMENT
 * @see Section.MAP
 */
public class Section extends ArrayList<Section.ELEMENT> implements Cloneable {

    /**
    * Represents the elements contained in a section.
    */
    public static class ELEMENT extends ArrayList<Expression>
                                implements Cloneable {

        private static final long serialVersionUID = 13L;

        private static final Expression one = new IntegerLiteral(1);

        /**
        * Constructs an empty element.
        */
        public ELEMENT() {
            super();
        }

        /**
        * Constructs a new element from the given array access.
        *
        * @param acc the array access from which the new element is constructed.
        */
        public ELEMENT(ArrayAccess acc) {
            for (int i = 0; i < acc.getNumIndices(); ++i) {
                add(Symbolic.simplify(acc.getIndex(i)));
            }
        }

        /**
        * Returns a clone of this element.
        *
        * @return the cloned object.
        */
        @Override
        public ELEMENT clone() {
            ELEMENT o = new ELEMENT();
            for (int i = 0; i < this.size(); i++) {
                o.add(this.get(i).clone());
            }
            return o;
        }

        /**
        * Checks if this element is equal to the specified object.
        *
        * @param o the object to be compared with.
        */
        public boolean equals(Object o) {
            if (o == null || o.getClass() != this.getClass()) {
                return false;
            }
            ELEMENT other = (ELEMENT)o;
            if (size() != other.size()) {
                return false;
            }
            for (int i = 0; i < this.size(); ++i) {
                if (!this.get(i).equals(other.get(i))) {
                    return false;
                }
            }
            return true;
        }

        // Checks if this element is adjacent west to the other element in the
        // specified dimension while other subscripts are identical in the other
        // dimensions. [0][2:5] isAdjacentWest to [1][2:5].
        private boolean isAdjacentWest(ELEMENT other, int loc) {
            return (isAdjacentTo(other, loc) == -1);
        }

        // Checks if this element is adjacent east to the other element.
        private boolean isAdjacentEast(ELEMENT other, int loc) {
            return (isAdjacentTo(other, loc) == 1);
        }

        // Base check method for isAdjacent method.
        private int isAdjacentTo(ELEMENT other, int loc) {
            if (other == null || size() != other.size()) {
                return 0;
            }
            // All subscripts in the other dimensions should be equal to the
            // other elements' subscripts.
            for (int i = 0; i < size(); ++i) {
                if (i != loc && !get(i).equals(other.get(i))) {
                    return 0;
                }
            }
            RangeExpression re1 = RangeExpression.toRange(get(loc));
            RangeExpression re2 = RangeExpression.toRange(other.get(loc));
            Expression e12 = Symbolic.subtract(re2.getLB(), re1.getUB());
            Expression e21 = Symbolic.subtract(re1.getLB(), re2.getUB());
            if (e12.equals(one)) {
                return -1; // UB(e1)+1 = LB(e2)
            } else if (e21.equals(one)) {
                return 1;  // UB(e2)+1 = LB(e1);
            } else {
                return 0;
            }
        }

        // Check if this element encloses the other.
        private boolean enclose(ELEMENT other, RangeDomain rd) {
            for (int i = 0; i < size(); i++) {
                RangeExpression re0 = RangeExpression.toRange(get(i));
                RangeExpression re1 = RangeExpression.toRange(other.get(i));
                Expression lb0 = re0.getLB(), ub0 = re0.getUB();
                Expression lb1 = re1.getLB(), ub1 = re1.getUB();
                if (rd.isLE(lb0, lb1) && rd.isGE(ub0, ub1)) {
                    ; // it encloses !!!
                } else {
                    return false;
                }
            }
            return true;
        }

        /**
        * Converts this element to a string.
        *
        * @return the string representation of this element.
        */
        public String toString() {
            StringBuilder str = new StringBuilder(80);
            str.append("[");
            for (int i = 0; i < size(); i++) {
                Expression e = get(i);
                if (i > 0) {
                    str.append("][");
                }
                if (e instanceof RangeExpression) {
                    RangeExpression re = (RangeExpression)e;
                    str.append(re.getLB()).append(":").append(re.getUB());
                } else {
                    str.append(e);
                }
            }
            str.append("]");
            return str.toString();
        }

        /**
        * Performs intersection operation between two section elements with the
        * specified range domain.
        *
        * @param other the other element.
        * @param rd the specified range domain.
        * @return the result of the intersection.
        */
        public ELEMENT intersectWith(ELEMENT other, RangeDomain rd) {
            ELEMENT ret = new ELEMENT();
            for (int i = 0; i < size(); ++i) {
                Expression intersected =
                        intersectBound(get(i), other.get(i), rd);
                if (intersected == null) { // Either it is empty or unknown
                    return null;
                }
                ret.add(intersected);
            }
            return ret;
        }

        /**
        * Performs union operation between two section elements with the
        * specified range domain.
        *
        * @param other the other element.
        * @param rd the specified range domain.
        * @return the result of the union.
        */
        public ELEMENT unionWith(ELEMENT other, RangeDomain rd) {
            ELEMENT ret = new ELEMENT();
            for (int i = 0; i < size(); ++i) {
                Expression unioned = unionBound(get(i), other.get(i), rd);
                if (unioned == null) { // Either it has holes or unknown
                    return null;
                }
                ret.add(unioned);
            }
            return ret;
        }

        /**
        * Performs difference operation between two section elements with the
        * specified range domain.
        *
        * @param other the other element.
        * @param rd the specified range domain.
        * @return the resulting section of the difference.
        */
        public Section differenceFrom(ELEMENT other, RangeDomain rd) {
            // Temporary list containing the result of differences for each
            // dimension
            Section ret = new Section(size());
            // Process easy case: other encloses this.
            if (other.enclose(this, rd)) {
                return ret;
            }
            for (int i = 0; i < size(); ++i) {
                Expression expr = this.get(i);
                List<Expression> temp_i = new ArrayList<Expression>(size());
                Expression intersected = intersectBound(expr, other.get(i), rd);
                if (intersected == null) {
                    temp_i.add(expr.clone());
                } else {
                    RangeExpression re_inct =
                            RangeExpression.toRange(intersected);
                    RangeExpression re_from = RangeExpression.toRange(expr);
                    Expression left_ub = Symbolic.subtract(re_inct.getLB(),one);
                    Expression right_lb = Symbolic.add(re_inct.getUB(), one);
                    Relation rel = rd.compare(re_from.getLB(), left_ub);
                    if (!rel.isGT()) {
                        temp_i.add((new RangeExpression(re_from.getLB().clone(),
                                left_ub)).toExpression());
                    }
                    rel = rd.compare(right_lb, re_from.getUB());
                    if (!rel.isGT()) {
                        temp_i.add((new RangeExpression(right_lb,
                                re_from.getUB().clone())).toExpression());
                    }
                }
                for (int j = 0; j < temp_i.size(); j++) {
                    ELEMENT new_section = this.clone();
                    new_section.set(i, temp_i.get(j));
                    ret.add(new_section);
                }
            }
            return ret;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    private static final long serialVersionUID = 12L;

    // Dimension
    private int dimension;

    /**
    * Constructs a section with the specified dimension.
    *
    * @param dimension the dimension of the array. -1 for scalar variables.
    */
    public Section(int dimension) {
        super();
        this.dimension = dimension;
    }

    /**
    * Constructs a section with the specified array access.
    *
    * @param acc the array access expression.
    */
    public Section(ArrayAccess acc) {
        this(acc.getNumIndices());
        add(new ELEMENT(acc));
    }

    /**
    * Clones a section object.
    *
    * @return the cloned section.
    */
    @Override
    public Section clone() {
        Section o = new Section(dimension);
        // Make a deep copy since ArrayList makes only a shallow copy.
        for (int i = 0; i < this.size(); i++) {
            o.add(this.get(i).clone());
        }
        return o;
    }

    /**
    * Adds a new element in the section.
    *
    * @param elem the new element to be added.
    * @return true (as per the general contract of Collection.add).
    */
    public boolean add(ELEMENT elem) {
        if (!contains(elem)) {
            super.add(elem);
        }
        return true;
    }

    /**
    * returns a dimension
    */
    public int getDimension() {
        return dimension;
    }

    /**
    * Checks if the section is for a scalar variable.
    */
    public boolean isScalar() {
        return (isEmpty() && dimension == -1);
    }

    /**
    * Checks if the section is for an array variable.
    */
    public boolean isArray() {
        return (dimension > 0);
    }

    /**
    * Checks if the section contains the specified variables.
    */
    public boolean containsSymbols(Set<Symbol> vars) {
        for (int i = 0; i < this.size(); i++) {
            ELEMENT elem = this.get(i);
            for (int j = 0; j < elem.size(); j++) {
                if (IRTools.containsSymbols(elem.get(j), vars)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
    * Expand every section under the constraints given by the range domain.
    *
    * @param rd the given range domain.
    * @param vars the set of symbols to be expanded.
    */
    public void expandMay(RangeDomain rd, Set<Symbol> vars) {
        PrintTools.printlnStatus(5, "[Section.expandMay]", "vars =",
                PrintTools.collectionToString(vars, ","));
        PrintTools.printlnStatus(5, "[Section.expandMay]", "rd =", rd);
        for (int i = 0; i < this.size(); i++) {
            ELEMENT elem = this.get(i);
            for (int j = 0; j < dimension; ++j) {
                Expression e = elem.get(j);
                PrintTools.printlnStatus(5,
                        "[Section.expandMay]", "section =", e);
                e = rd.expandSymbols(e, vars);
                PrintTools.printlnStatus(5,
                        "[Section.expandMay]", "expanded =", e);
                elem.set(j, e);
            }
        }
        simplifyMe();
    }

    /**
    * Expand every section under the constraints given by the range domain.
    *
    * @param rd the given range domain.
    * @param ivs the set of symbols to be expanded.
    * @param vars the set of symbols that should not be part of the expansion.
    */
    public void expandMust(RangeDomain rd, Set<Symbol> ivs, Set<Symbol> vars) {
        for (int i = 0; i < this.size(); i++) {
            ELEMENT elem = this.get(i);
            for (int j = 0; j < dimension; j++) {
                Expression e = rd.expandSymbols(elem.get(j), ivs);
                if (e == null ||
                    (e instanceof RangeExpression &&
                            !((RangeExpression)e).isBounded()) ||
                    IRTools.containsSymbols(e, vars)) {
                    this.remove(i--);
                    break;
                }
                elem.set(j, e);
            }
        }
        simplifyMe();
    }

    /**
    * Substitutes any variables having symbolic constant values in the section
    * avoiding cases that produces expression having a variable in the
    * "avoid" set.
    */
    public void substituteForward(RangeDomain rd, Set<Symbol> avoid) {
        for (int i = 0; i < this.size(); i++) {
            ELEMENT elem = this.get(i);
            for (int j = 0; j < dimension; j++) {
                Expression e = rd.substituteForward(elem.get(j));
                if (!IRTools.containsSymbols(e, avoid)) {
                    elem.set(j, e);
                }
            }
        }
    }

    /**
    * Performs intersection operation between two sections with the specified
    * range domain.
    *
    * @param other the section being intersected with.
    * @param rd the supporting range domain.
    * @return the resulting section.
    */
    public Section intersectWith(Section other, RangeDomain rd) {
        // No intersection is possible; returns null (check at the higher level)
        if (dimension != other.dimension) {
            return null;
        }
        Section ret = new Section(dimension);
        for (int i = 0; i < this.size(); i++) {
            for (int j = 0; j < other.size(); j++) {
                ELEMENT elem = this.get(i).intersectWith(other.get(j), rd);
                if (elem != null ){
                    ret.add(elem);
                }
            }
            if (ret.isEmpty()) {
                break;
            }
        }
        ret = ret.simplify();
        PrintTools.printlnStatus(5, this, "(^)", other, "=", ret, "under", rd);
        return ret;
    }

    /**
    * Performs union operation between two sections with the specified range
    * domain.
    *
    * @param other the section being unioned with.
    * @param rd the supporting range domain.
    * @return the resulting section.
    */
    public Section unionWith(Section other, RangeDomain rd) {
        if (dimension != other.dimension) {
            return null;
        }
        Section ret = new Section(dimension);
        int i, j;
        for (i = 0, j = 0; i < this.size() || j < other.size(); i++, j++) {
            if (i >= this.size()) {
                ret.add(other.get(j).clone());
            } else if (j >= other.size()) {
                ret.add(this.get(i).clone());
            } else {
                ELEMENT elem1 = this.get(i);
                ELEMENT elem2 = other.get(j);
                ELEMENT unioned = elem1.unionWith(elem2, rd);
                if (unioned == null) {
                    ret.add(elem1.clone());
                    ret.add(elem2.clone());
                } else {
                    ret.add(unioned);
                }
            }
        }
        ret = ret.simplify();
        PrintTools.printlnStatus(5, this, "(v)", other, "=", ret, "under", rd);
        return ret;
    }


    /**
    * Performs difference operation between two sections with the specified
    * range domain.
    *
    * @param other the other section from which this section is differenced.
    * @param rd the supporting range domain.
    * @return the resulting section.
    */
    public Section differenceFrom(Section other, RangeDomain rd) {
        Section ret = this.clone();
        // Just return a clone upon dimension mismatch
        if (dimension != other.dimension) {
            PrintTools.printlnStatus(5, this, "(-)", other, "=", ret);
            return ret;
        }
        for (int i = 0; i < other.size(); i++) {
            Section curr = new Section(dimension);
            for (int j = 0; j < ret.size(); j++) {
                Section diffed = ret.get(j).differenceFrom(other.get(i), rd);
                for (int k = 0; k < diffed.size(); k++) {
                    //if (!curr.contains(diffed.get(k)))
                    curr.add(diffed.get(k));
                }
            }
            ret = curr;
        }
        ret = ret.simplify();
        PrintTools.printlnStatus(5, this, "(-)", other, "=", ret, "under", rd);
        return ret;
    }


    /**
    * Returns union of two symbolic bounds
    */
    private static Expression
            unionBound(Expression e1, Expression e2, RangeDomain rd) {
        Expression intersected = intersectBound(e1, e2, rd);
        //System.out.println("intersected = "+intersected);
        if (intersected == null) {  // Either it has no intersection or unknown.
            return null;            // Merging i,i+1 => i:i+1 is disregarded
        }                           // for now.
        RangeExpression re1 = RangeExpression.toRange(e1);
        RangeExpression re2 = RangeExpression.toRange(e2);
        Expression lb = null, ub = null;
        Relation rel = rd.compare(re1.getLB(), re2.getLB());
        if (rel.isLE()) {
            lb = re1.getLB();
        } else if (rel.isGE()) {
            lb = re2.getLB();
        } else {
            return null;
        }
        rel = rd.compare(re1.getUB(), re2.getUB());
        if (rel.isGE()) {
            ub = re1.getUB();
        } else if (rel.isLE()) {
            ub = re2.getUB();
        } else {
            return null;
        }
        return (new RangeExpression(lb.clone(), ub.clone())).toExpression();
    }

    /**
    * Returns intersection of two symbolic intervals
    */
    private static Expression
            intersectBound(Expression e1, Expression e2, RangeDomain rd) {
        RangeExpression re1 = RangeExpression.toRange(e1);
        RangeExpression re2 = RangeExpression.toRange(e2);
        Expression lb = null, ub = null;
        Relation rel = rd.compare(re1.getLB(), re2.getLB());
        if (rel.isGE()) {
            lb = re1.getLB();
        } else if (rel.isLE()) {
            lb = re2.getLB();
        } else {
            return null;
        }
        rel = rd.compare(re1.getUB(), re2.getUB());
        if (rel.isLE()) {
            ub = re1.getUB();
        } else if (rel.isGE()) {
            ub = re2.getUB();
        } else {
            return null;
        }
        // Final check if lb>ub.
        rel = rd.compare(lb, ub);
        if (!rel.isLE()) {
            return null;
        } else {
            return (new RangeExpression(lb.clone(), ub.clone())).toExpression();
        }
        /* temporary fix for the replacement of the above commented section */
        /* use this fix with caution - it may break many innocent things.
        return (new RangeExpression(lb, ub)).toExpression();
        */
    }


    /**
    * Removes section elements that contain the specified variable.
    */
    public void removeAffected(Symbol var) {
        for (int i = 0; i < this.size(); i++) {
            ELEMENT elem = this.get(i);
            for (int j = 0; j < elem.size(); j++) {
                if (IRTools.containsSymbol(elem.get(j), var)) {
                    this.remove(i--);
                    break;
                }
            }
        }
    }

    /**
    * Removes section elements that is affected by the specified function call.
    */
    public void removeSideAffected(FunctionCall fc) {
        Set<Symbol> params = SymbolTools.getAccessedSymbols(fc);
        for (int i = 0; i < this.size(); i++) {
            ELEMENT elem = this.get(i);
            boolean kill = false;
            for (int j = 0; j < elem.size(); j++) {
                Set<Symbol> vars = SymbolTools.getAccessedSymbols(elem.get(j));
                // Case 1: variables in section representation are used as
                // parameters.
                if (IRTools.containsSymbols(vars, params)) {
                    kill = true;
                    break;
                }
                // Case 2: variables in section representation are global.
                for (Symbol var : vars) {
                    if (SymbolTools.isGlobal(var, fc)) {
                        kill = true;
                        break;
                    }
                }
                if (kill) {
                    break;
                }
            }
            if (kill) {
                this.remove(i--);
            }
        }
    }

    /**
    * Converts this section to a string.
    *
    * @return the string representation of the section.
    */
    public String toString() {
        return ("{" + PrintTools.listToString(this, ", ") + "}");
    }

    // Simplify the adjacent section elements.
    private Section simplify() {
        if (dimension < 1) {
            return this;
        }
        Section ret = null;
        // Adjacenct elements. [1][0],[1][1] => [1][0:1]
        for (int i = 0; i < dimension; ++i) {
            Section temp = (ret==null)? this.clone(): ret;
            ret = new Section(dimension);
            while (!temp.isEmpty()) {
                ELEMENT elem1 = temp.remove(0);
                for (int j = 0; j < temp.size(); j++) {
                    ELEMENT elem2 = temp.get(j);
                    if (elem2.isAdjacentWest(elem1, i)) {
                        RangeExpression re1 =
                                RangeExpression.toRange(elem1.get(i));
                        RangeExpression re2 =
                                RangeExpression.toRange(elem2.get(i));
                        elem1.set(i, new RangeExpression(
                                re2.getLB().clone(), re1.getUB().clone()));
                        temp.remove(j--);
                    } else if (elem2.isAdjacentEast(elem1, i)) {
                        RangeExpression re1 =
                                RangeExpression.toRange(elem1.get(i));
                        RangeExpression re2 =
                                RangeExpression.toRange(elem2.get(i));
                        elem1.set(i, new RangeExpression(
                                re1.getLB().clone(), re2.getUB().clone()));
                        temp.remove(j--);
                    }
                }
                ret.add(elem1);
            }
        }
        // Enclosed elements. [1][0:1],[1][0] => [1][0:1]
        RangeDomain rd = new RangeDomain();
        Section temp = ret.clone();
        ret.clear();
        while (!temp.isEmpty()) {
            ELEMENT elem1 = temp.remove(0);
            Iterator<ELEMENT> iter = temp.iterator();
            for (int i = 0; i < temp.size(); i++) {
                ELEMENT elem2 = temp.get(i);
                if (elem2.enclose(elem1, rd)) {
                    elem1 = elem2;
                    temp.remove(i--);
                } else if (elem1.enclose(elem2, rd)) {
                    temp.remove(i--);
                }
            }
            ret.add(elem1);
        }
        return ret;
    }

    // In-place simplification.
    private void simplifyMe() {
        Section simplified = this.simplify();
        this.clear();
        this.addAll(simplified);
    }

    // Returns predicates from the unresolved set of bounds from the section.
    private RangeDomain getPredicates(RangeDomain rd)
    {
        RangeDomain ret = new RangeDomain();
        if (!isArray()) {
            return ret;
        }
        for (int i = 0; i < this.size(); i++) {
            ELEMENT elem = this.get(i);
            for (int j = 0; j < elem.size(); j++) {
                Expression e = elem.get(j);
                if (e instanceof RangeExpression) {
                    RangeExpression re = (RangeExpression)e;
                    Relation rel = rd.compare(re.getLB(), re.getUB());
                    if (!rel.isLE()) {
                        ret.intersectRanges(RangeAnalysis.extractRanges(
                                Symbolic.le(re.getLB(), re.getUB())));
                    }
                }
            }
        }
        return ret;
    }

    ////////////////////////////////////////////////////////////////////////////

    /**
    * Class MAP represents map from variables to their sections. For the
    * convenience of implementation, we assign empty section for scalar
    * variables.
    */
    public static class MAP extends HashMap<Symbol, Section>
            implements Cloneable {

        private static final long serialVersionUID = 14L;

        /**
        * Constructs an empty map.
        */
        public MAP() {
            super();
        }

        /**
        * Constructs a map with a pair of variable and section.
        *
        * @param var the key variable.
        * @param section the section associated with the variable.
        */
        public MAP(Symbol var, Section section) {
            super();
            put(var, section);
        }

        /**
        * Returns a clone object.
        */
        @Override
        public MAP clone() {
            MAP o = new MAP();
            for (Symbol var : keySet()) {
                o.put(var, get(var).clone());
            }
            return o;
        }

        /**
        * Cleans up empty sections.
        */
        public void clean() {
            Set<Symbol> vars = new HashSet<Symbol>(keySet());
            for (Symbol var : vars) {
                Section sec = this.get(var);
                if (var == null || sec.dimension > 0 && sec.isEmpty()) {
                    this.remove(var);
                }
            }
        }

        /**
        * Performs intersection operation between the two section maps with the
        * specified range domain.
        *
        * @param other the other section map to be intersected with.
        * @param rd the specified range domain.
        * @return the resulting section map after intersection.
        */
        public MAP intersectWith(MAP other, RangeDomain rd) {
            MAP ret = new MAP();
            if (other == null) {
                return ret;
            }
            for (Symbol var : keySet()) {
                Section s1 = this.get(var);
                Section s2 = other.get(var);
                if (s1 == null || s2 == null) {
                    continue;
                }
                if (s1.isScalar() && s2.isScalar()) {
                    ret.put(var, s1.clone());
                } else {
                    Section intersected = s1.intersectWith(s2, rd);
                    if (intersected == null) {
                        PrintTools.printlnStatus(0,
                                "[WARNING] Dimension mismatch");
                    } else {
                        ret.put(var, intersected);
                    }
                }
            }
            ret.clean();
            return ret;
        }

        /**
        * Performs union operation between the two section maps with the
        * specified range domain.
        *
        * @param other the other section map to be united with.
        * @param rd the specified range domain.
        * @return the resulting section map after union.
        */
        public MAP unionWith(MAP other, RangeDomain rd) {
            if (other == null) {
                return this.clone();
            }
            MAP ret = new MAP();
            Set<Symbol> vars = new HashSet<Symbol>(keySet());
            vars.addAll(other.keySet());
            for (Symbol var : vars) {
                Section s1 = this.get(var);
                Section s2 = other.get(var);
                if (s1 == null && s2 == null) {
                    continue;
                }
                if (s1 == null) {
                    ret.put(var, s2.clone());
                } else if (s2 == null) {
                    ret.put(var, s1.clone());
                } else if (s1.isScalar() && s2.isScalar()) {
                    ret.put(var, s1.clone());
                } else {
                    Section unioned = s1.unionWith(s2, rd);
                    if (unioned == null) {
                        ret.put(var, s2.clone()); // heuristics - second operand
                    } else {
                        ret.put(var, unioned);
                    }
                }
            }
            ret.clean();
            return ret;
        }

        /**
        * Performs difference operation between the two section maps with the
        * specified range domain.
        *
        * @param other the other section map to be differenced from.
        * @param rd the specified range domain.
        * @return the resulting section map after difference.
        */
        public MAP differenceFrom(MAP other, RangeDomain rd) {
            if (other == null) {
                return this.clone();
            }
            MAP ret = new MAP();
            Set<Symbol> vars = new HashSet<Symbol>(keySet());
            for (Symbol var : vars) {
                Section s1 = this.get(var);
                Section s2 = other.get(var);
                if (s2 == null) {
                    ret.put(var, s1.clone());
                } else if (s1.isArray() || s2.isArray()) {
                    ret.put(var, s1.differenceFrom(s2, rd));
                }
            }
            ret.clean();
            return ret;
        }

        /**
        * Performs conditional difference operation after adding unresolved
        * bound relation from the current section map. This operation enhances
        * the coverage of difference operation if there exist some bound
        * expressions that are not guaranteed to be true (lb&lt;=ub). This
        * should be used only for upward-exposed set computation.
        */
        public MAP differenceFrom2(MAP other, RangeDomain rd) {
            if (other == null) {
                return this.clone();
            }
            MAP ret = new MAP();
            Set<Symbol> vars = new HashSet<Symbol>(keySet());
            for (Symbol var : vars) {
                Section s1 = this.get(var), s2 = other.get(var);
                if (s2 == null) {
                    ret.put(var, s1.clone());
                } else if (s1.isArray() || s2.isArray()) {
                    RangeDomain modified = rd.clone();
                    RangeDomain predicates = s1.getPredicates(rd);
                    for (Symbol symbol : predicates.getSymbols()) {
                        modified.setRange(symbol, predicates.getRange(symbol));
                    }
                    ret.put(var, s1.differenceFrom(s2, modified));
                }
            }
            ret.clean();
            return ret;
        }

        /**
        * Removes sections that contains the specified symbol.
        */
        public void removeAffected(Symbol var) {
            Set<Symbol> keys = new HashSet<Symbol>(keySet());
            for (Symbol key : keys) {
                this.get(key).removeAffected(var);
            }
            clean();
        }

        /**
        * Removes sections that contains the specified set of variables.
        */
        public void removeAffected(Collection<Symbol> vars) {
            for (Symbol var : vars) {
                removeAffected(var);
            }
        }

        /**
        * Removes sections that are unsafe in the given traversable object due
        * to function calls.
        */
        public void removeSideAffected(Traversable tr) {
            DFIterator<FunctionCall> iter =
                    new DFIterator<FunctionCall>(tr, FunctionCall.class);
            iter.pruneOn(FunctionCall.class);
            while (iter.hasNext()) {
                FunctionCall fc = iter.next();
                Set<Symbol> vars = new HashSet<Symbol>(keySet());
                for (Symbol var : vars) {
                    this.get(var).removeSideAffected(fc);
                }
                clean();
            }
        }

        public void print(String str) {
            print(str, 7);
        }

        public void print(String str, int verbosity) {
            if (isEmpty()) {
                PrintTools.println(str + " is empty", verbosity);
            } else {
                if (keySet() == null) {
                    PrintTools.println(str + " keySet() is null", verbosity);
                } else if (keySet().size() == 0) {
                    PrintTools.println(str + " keySet() is empty", verbosity);
                } else {
                    PrintTools.print(str + " = [", verbosity);
                    int count=0;
                    for (Symbol symbol : keySet()) {
                        if (++count > 1) PrintTools.print(", ", verbosity);
                        Section section = get(symbol);
                        if (section.getDimension() < 1) {   // scalar variable
                            PrintTools.print(symbol.getSymbolName(), verbosity);
                        } else {
                            PrintTools.print(symbol.getSymbolName() + "=" +
                                    section.toString(), verbosity);  
                        }
                    }
                    PrintTools.println("]", verbosity);
                }
            }
        }

        @Override
        public String toString() {
            StringBuilder str = new StringBuilder(80);
            str.append("<");
            int count = 0;
            for (Symbol symbol : keySet()) {
                if (count++ > 0) {
                    str.append(", ");
                }
                Section section = this.get(symbol);
                str.append(symbol.getSymbolName()).append(" => ");
                if (section.getDimension() < 1) {  // scalar variable
                    str.append("[]");
                } else {
                     str.append(section);
                }
            }
            str.append(">");
            return str.toString();
        }
    }
}
