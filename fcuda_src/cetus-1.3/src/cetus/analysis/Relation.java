package cetus.analysis;

/**
 * Class Relation represents symbolic relationship between two expressions.
 * The result of a symbolic comparison method in {@link RangeDomain} returns a
 * Relation object.
 */
public class Relation {

    // partial relations 
    private boolean lt, eq, gt, circular;

    /**
     * Constructs an empty relation that implies unknown relationship.
     */
    public Relation() {
        lt = false; eq = false; gt = false; circular = false;
    }

    /**
     * Constructs a relation object with the specified set of relationships.
     *
     * @param lt the "less than" relation.
     * @param eq the "equal to" relation.
     * @param gt the "greater than" relation.
     */
    public Relation(boolean lt, boolean eq, boolean gt) {
        this.lt = lt; this.eq = eq; this.gt = gt;
        this.circular = false;
    }

    public void setLT(boolean lt) {
        this.lt = lt;
    }
    
    public void setEQ(boolean eq) {
        this.eq = eq;
    }

    public void setGT(boolean gt) {
        this.gt = gt;
    }

    public void setCircular(boolean circular) {
        this.circular = circular;
    }

    /**
     * Checks if "less than" is implied in this relation.
     */
    public boolean isLT() {
        return (lt && !eq && !gt);
    }

    /**
     * Checks if "less equal" is implied in this relation.
     */
    public boolean isLE() {
        return ((lt || eq) && !gt);
    }

    /**
     * Checks if "equal to" is implied in this relation.
     */
    public boolean isEQ() {
        return (!lt && eq && !gt);
    }

    /**
     * Checks if "not equal" is implied in this relation.
     */
    public boolean isNE() {
        return (!eq && (lt || gt));
    }

    /**
     * Checks if "greater than" is implied in this relation.
     */
    public boolean isGT() {
        return (!lt && !eq && gt);
    }

    /**
     * Checks if "greater equal" is implied in this relation.
     */
    public boolean isGE() {
        return (!lt && (eq || gt));
    }

    /**
     * Checks if there is no known relationship.
     */
    public boolean isUnknown() {
        return ((!lt && !eq && !gt) || (lt && eq && gt));
    }

    public boolean isCircular() {
        return circular;
    }

    /**
     * Logical AND operation between two relation objects.
     */
    public static Relation AND(Relation rel1, Relation rel2) {
        Relation ret = new Relation(
            rel1.lt && rel2.lt,
            rel1.eq && rel2.eq,
            rel1.gt && rel2.gt
        );
        ret.setCircular(rel1.circular && rel2.circular);
        return ret;
    }

    /**
     * Logical OR operation between two relation objects.
     */
    public static Relation OR(Relation rel1, Relation rel2) {
        Relation ret = new Relation(
            rel1.lt || rel2.lt,
            rel1.eq || rel2.eq,
            rel1.gt || rel2.gt
        );
        ret.setCircular(rel1.circular || rel2.circular);
        return ret;
    }

    /**
     * Returns the string representation of the relation.
     */
    public String toString() {
        String ret = new String();
        if (isEQ()) {
            ret += "==";
        } else if (isLT()) {
            ret += "<";
        } else if (isGT()) {
            ret += ">";
        } else if (isLE()) {
            ret += "<=";
        } else if (isGE()) {
            ret += ">=";
        } else if (isNE()) {
            ret += "!=";
        } else {
            ret += "?";
        }
        if (isCircular()) {
            ret += "@";
        }
        return ret;
    }
}
