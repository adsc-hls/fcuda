package cetus.analysis;

import cetus.hir.Symbol;

/**
 * Points to relationships represent the link from the 
 * head of the relationship to the tail. The head is the 
 * pointer variable/symbol and the tail is the variable/location/
 * symbol being pointed to. The relationship can be 
 * definite or possible.
 */
public class PointsToRel implements Cloneable {

    // Points-to relationship pair
    // Head
    private Symbol pointer;
    // Tail
    private Symbol points_to;
    // Type of relationship
    private boolean definite;

    /**
     * Constructor
     * @param head Symbol for head location, always a pointer
     * @param tail Symbol for tail location
     * @param type The definiteness of the relationship
     */
    public PointsToRel(Symbol head, Symbol tail, boolean type) {
        this.pointer = head;
        this.points_to = tail;
        this.definite = type;
    }

    /**
     * Constructor - default false relationship
     * @param head Symbol for head location, always a pointer
     * @param tail Symbol for tail location
     */
    public PointsToRel(Symbol head, Symbol tail) {
        this(head, tail, false);
    }

    @Override
    public PointsToRel clone() {
        PointsToRel clone = new PointsToRel(pointer, points_to, definite);
        return clone;
    }
    
    @Override
    public int hashCode() {
        int ret = pointer.hashCode();
        ret = 31*ret + points_to.hashCode();
        ret = 31*ret + ((definite)? 'D' : 'P');
        return ret;
        //return (pointer.hashCode() + points_to.hashCode() + ((definite)? 1: 0));
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || !(o instanceof PointsToRel)) {
            return false;
        }
        PointsToRel other = (PointsToRel)o;
        return (this.pointer.equals(other.pointer) &&
                this.points_to.equals(other.points_to) &&
                this.definite == other.definite);
    }
    
    /**
     * Return the head of the relationship
     */
    public Symbol getPointerSymbol() {
        return this.pointer;
    }

    /**
     * Return the tail of the relationship
     */
    public Symbol getPointedToSymbol() {
        return this.points_to;
    }

    /**
     * Is the relationship definitely valid
     */
    public boolean isDefinite() {
        return this.definite;
    }

    /**
     * Set to definite
     */
    public void setDefinite() {
        this.definite = true;
    }

    /**
     * Set to possible
     */
    public void setPossible() {
        this.definite = false;
    }

    /**
     * Merge this rel with the input rel. Handles 
     * merging the relationship types.
     * @param rel input relationship
     * @return new merged relationship
     */
    public PointsToRel mergeRel(PointsToRel rel) {
        PointsToRel return_rel = new PointsToRel(pointer, points_to);
        if (this.definite && rel.definite) {
            return_rel.definite = true;
        }
        return return_rel;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        sb.append("(").append(pointer.getSymbolName());
        sb.append(",").append(points_to.getSymbolName()).append(",");
        if (definite) {
            sb.append("D");
        } else {
            sb.append("P");
        }
        sb.append(")");
        return sb.toString();
    }
}
