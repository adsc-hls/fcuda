package cetus.analysis;

import cetus.hir.Symbol;

import java.util.*;

/**
 * PointsToDomain represents a set of pointer-relationships at 
 * a given program point, and is used by the Points-to Analyzer.
 * <p>
 * Implements the domain interface used by Cetus analysis 
 * passes for easier flexibility and sharing between passes. 
 * Specifically implements the merge, union, diff, intersect, 
 * kill and clone functionality.
 * <p>
 * Relationships inside of this domain are represented by 
 * {@link PointsToRel}.
 * 
 * @see PointsToRel
 * @see Domain
 */
public class PointsToDomain extends LinkedHashMap<Symbol,HashSet<PointsToRel>> 
                            implements Domain {

    // Just to prevent javac from complaining.
    private static final long serialVersionUID = 15L;

    /**
     * Create an empty Domain constructor
     */
    public PointsToDomain() {
        super();
    }

    /**
     * Create a domain with a single pointer relationship
     * @param p_rel
     */
    public PointsToDomain(PointsToRel p_rel) {
        super();
        addRel(p_rel);
    }

    /**
     * Provides deep clone
     */
    @Override
    public PointsToDomain clone() {
        PointsToDomain cloned = new PointsToDomain();
        for (Symbol s : this.keySet()) {
            HashSet<PointsToRel> new_rels = new LinkedHashSet<PointsToRel>();
            for (PointsToRel rel : this.get(s)) {
                new_rels.add(rel.clone());
            }
            cloned.put(s, new_rels);
        }
        return cloned;
    }

    /**
     * Compares equality based on set comparison for 
     * set of PointsToRel
     */
    public boolean equals(Domain other) {
        boolean is_equal = false;
        if (other instanceof PointsToDomain) {
            Set<Symbol> s1_set = this.keySet();
            Set<Symbol> s2_set = ((PointsToDomain)other).keySet();
            // compare their keysets, if equal, test equality of
            // points to relationships for each symbol
            if (s1_set.isEmpty() && s2_set.isEmpty()) {
                is_equal = true;
            } else if (s1_set.equals(s2_set)) {
                for (Symbol s1 : s1_set) {
                    HashSet rel1_set = this.get(s1);
                    HashSet rel2_set = ((PointsToDomain)other).get(s1);
                    if (rel1_set.equals(rel2_set)) {
                        is_equal = true;
                    } else {
                        is_equal = false;
                        break;
                    }
                }            
            }
        }        
        return is_equal;
    }

    /**
     * Add relationship to this domain
     * @param p_rel Relationship to be added
     */
    public void addRel(PointsToRel p_rel) {
        HashSet<PointsToRel> value = this.remove(p_rel.getPointerSymbol());
        if (value == null) {
            value = new LinkedHashSet<PointsToRel>();
        }
        value.add(p_rel);
        put(p_rel.getPointerSymbol(), value);
    }

    /**
     * Union two points-to domains. This functionality is used 
     * to combine points-to domains that may contain the same 
     * information, additional information is then combined 
     * together. Different from merge functionality.
     * 
     * @param other Domain to union with
     * @return a new Domain which represents the union
     */
    public Domain union(Domain other) {
        Domain ret = this.clone(); 
        if (other instanceof PointsToDomain) {
            PointsToDomain other_ptd = (PointsToDomain)other;
            if (other_ptd.isEmpty()) {
                return ret;
            }
            if (this.isEmpty()) {
                return other_ptd.clone();
            }
            for (Symbol s : other_ptd.keySet()) {
                if (this.containsKey(s)) {
                    HashSet<PointsToRel> p1_set = this.get(s);
                    HashSet<PointsToRel> p2_set = other_ptd.get(s);
                    HashSet<PointsToRel> merged_set =
                            new LinkedHashSet<PointsToRel>();
                    merged_set.addAll(this.get(s));
                    merged_set.addAll(other_ptd.get(s));
                    for (PointsToRel p1 : p1_set) {
                        for (PointsToRel p2 : p2_set) {
                            Symbol head1 = p1.getPointerSymbol();
                            Symbol head2 = p2.getPointerSymbol();
                            Symbol tail1 = p1.getPointedToSymbol();
                            Symbol tail2 = p2.getPointedToSymbol();
                            // Found a pair of points to relationships with the
                            // same head and tail symbols
                            if (head1.equals(head2) && tail1.equals(tail2)) {
                                merged_set.remove(p1);
                                merged_set.remove(p2);
                                PointsToRel merged_p = p1.mergeRel(p2);
                                merged_set.add(merged_p);
                            }
                        }
                    }
                    HashSet<PointsToRel> merged_set_new = 
                            new LinkedHashSet<PointsToRel>();
                    for (PointsToRel rel : merged_set) {
                        PointsToRel new_rel = rel.clone();
                        merged_set_new.add(new_rel);
                    }
                    ((PointsToDomain)ret).remove(s);
                    ((PointsToDomain)ret).put(s, merged_set_new);
                } else {
                    PointsToDomain temp = other_ptd.clone();
                    ((PointsToDomain)ret).put(s, temp.get(s));
                }
            }
        } else if (other instanceof NullDomain) {
            // Do nothing, return this domain itself
        } else if (other instanceof Universe) {
            ret = other.union(this);
        }
        return ret;
    }

    /**
     * Merge this domain with another.
     * @param other Domain to merge with
     * @return merged domain
     */
    public Domain merge(Domain other) {
        return merge(this, other);
    }

    /**
     * Static method to merge two Points-to domains. Merge 
     * must handle types of relationships when combining 
     * the two domains, as opposed to a simple union. This 
     * functionality is used to combine information coming 
     * down different paths, and the new information 
     * must reflect the flow from more than one path.
     * @param s1 First domain in the merge
     * @param s2 Second domain in the merge
     * @return merged domain
     */
    public static Domain merge(Domain s1, Domain s2) {
        Domain merged = null;
        if (s1 instanceof PointsToDomain && s2 instanceof PointsToDomain) {
            PointsToDomain definite_set = new PointsToDomain();
            PointsToDomain possible_set = new PointsToDomain();
            // Obtain the definite set
            PointsToDomain intersection = (PointsToDomain)s1.intersect(s2);
            for (Symbol s : intersection.keySet()) {
                HashSet<PointsToRel> rel_set = intersection.get(s);
                for (PointsToRel p_rel : rel_set) {
                    if (p_rel.isDefinite()) {
                        definite_set.addRel(p_rel);
                    }
                }
            }
            // Obtain the possible set
            PointsToDomain union = (PointsToDomain)s1.union(s2);
            for (Symbol s : union.keySet()) {
                HashSet<PointsToRel> rel_set = union.get(s);
                HashSet<PointsToRel> definite_rels = definite_set.get(s);
                if (definite_rels == null) {
                    for(PointsToRel p_rel : rel_set) {
                        p_rel.setPossible();
                        possible_set.addRel(p_rel);
                    }
                } else {
                    for (PointsToRel p_rel : rel_set) {
                        Symbol tail = p_rel.getPointedToSymbol();
                        for (PointsToRel def_rel : definite_rels) {
                            if (!(tail.equals(def_rel.getPointedToSymbol()))) {
                                p_rel.setPossible();
                                possible_set.addRel(p_rel);
                            }
                        }
                    }
                }
            }
            // The union of the definite and possible sets is the merged set
            merged = (PointsToDomain)definite_set.union(possible_set);
        } else if (s1 instanceof Universe || s2 instanceof Universe) {
            merged = Universe.getUniverse();
        } else if (s1 instanceof NullDomain && s2 instanceof PointsToDomain) {
            merged = ((PointsToDomain)s2).clone();
        } else if (s1 instanceof PointsToDomain && s2 instanceof NullDomain) {
            merged = ((PointsToDomain)s1).clone();
        } else {
            merged = new NullDomain();
        }
        return merged;
    }

    /**
     * Extract all pointer relationships that are 
     * identical in the other points-to domain and 
     * return a domain representing the intersection
     */
    public Domain intersect(Domain other) {
        PointsToDomain ret = null;
        if (other instanceof PointsToDomain) {
            PointsToDomain other_ptd = (PointsToDomain)other;
            ret = new PointsToDomain();
            Set<Symbol> s1_set = other_ptd.keySet();
            for (Symbol s1 : s1_set) {
                HashSet<PointsToRel> rel1_set = other_ptd.get(s1);
                HashSet<PointsToRel> rel2_set = this.get(s1);
                if (rel1_set != null && rel2_set != null) {
                    for (PointsToRel p1_rel : rel1_set) {
                        for (PointsToRel p2_rel : rel2_set) {
                            if (p1_rel.equals(p2_rel)) {
                                // Found an intersecting relationship
                                PointsToRel intersect_rel = p1_rel.clone();
                                ret.addRel(intersect_rel);
                            }
                        }
                    }
                }
            }
        // other is Universe or NullDomain, the intersection is this current
        // points to domain
        } else {
            ret = this.clone();
        }
        return ret;
    }

    /**
     * Not supported
     */
    public Domain diffWeak(Domain other) {
        return null;
    }

    /**
     * Returns a subtraction of the relationships that are 
     * identical in the other domain.
     */
    public Domain diffStrong(Domain other) {
        PointsToDomain ret = null;
        if (other instanceof PointsToDomain) {
            PointsToDomain other_ptd = (PointsToDomain)other;
            ret = this.clone();
            Set<Symbol> keys = other_ptd.keySet();
            for (Symbol s : keys) {
                HashSet<PointsToRel> other_p_rel = other_ptd.get(s);
                HashSet<PointsToRel> this_p_rel = this.get(s);
                if (this_p_rel != null) {
                    for (PointsToRel p1_rel : other_p_rel) {
                        for (PointsToRel p2_rel : this_p_rel) {
                            if (p1_rel.equals(p2_rel)) {
                                HashSet<PointsToRel> ret_p_rel = ret.get(s);
                                ret_p_rel.remove(p2_rel);
                                if (ret_p_rel.isEmpty()) {
                                    ret.remove(s);
                                }
                            }
                        }
                    }
                }
                //((PointsToDomain)ret).remove(s);
            }
        }
        // If other is Universe, it contains points to 
        // for everything. Hence, diffStrong must produce
        // an empty Points to domain
        else if (other instanceof Universe) {
            ret = new PointsToDomain();
        } else {
            ret = this.clone();
        }
        return ret;
    }

    /**
     * Not supported
     */
    public void kill(Set<Symbol> vars) {
        ;// not supported, but need to be defined.
    }

    /**
     * Provides a new Points-to domain that contains all 
     * the pointer relationships for the symbols 
     * provided as input to the kill Set. The return 
     * domain relationships are the relationships that 
     * must be killed
     * @param exps Symbols whose relationships must be killed
     * @return the killed domain.
     */
    public Domain killSet(Set<Symbol> exps) {
        PointsToDomain kill_set = new PointsToDomain();
        for (Symbol s : this.keySet()) {
            if (exps.contains(s)) {
                kill_set.put(s, this.get(s));
            }
        }
        return kill_set;
    }

    public String toString2() {
        return super.toString();
    }

    @Override
    public String toString() {
        return PTDtoString(this);
    }

    // Alternative print methods for PointsToDomain.
    private static void SPTRtoString(Set<PointsToRel> rels, StringBuilder ret) {
        TreeSet<String> ordered_set = new TreeSet<String>();
        for (PointsToRel rel : rels) {
            ordered_set.add(PTRtoString(rel));
        }
        Iterator<String> iter = ordered_set.iterator();
        if (iter.hasNext()) {
            ret.append(iter.next());
            while (iter.hasNext()) {
                ret.append(", ").append(iter.next());
            }
        }
    }

    private static String PTRtoString(PointsToRel rel) {
        StringBuilder str = new StringBuilder(16);
        str.append("(").append(rel.getPointerSymbol().getSymbolName());
        str.append(",").append(rel.getPointedToSymbol().getSymbolName());
        if (rel.isDefinite()) {
            str.append(",D)");
        } else {
            str.append(",P)");
        }
        return str.toString();
    }

    private static String PTDtoString(PointsToDomain ptd) {
        TreeMap<String, Symbol> ordered = new TreeMap<String, Symbol>();
        for (Symbol key : ptd.keySet()) {
            ordered.put(key.getSymbolName(), key);
        }
        StringBuilder str = new StringBuilder(80);
        str.append("[");
        Iterator<String> iter = ordered.keySet().iterator();
        if (iter.hasNext()) {
            SPTRtoString(ptd.get(ordered.get(iter.next())), str);
            while (iter.hasNext()) {
                str.append(", ");
                SPTRtoString(ptd.get(ordered.get(iter.next())), str);
            }
        }
        str.append("]");
        return str.toString();
    }

    /**
    * Checks if the specified points-to relation is found.
    */
    public boolean containsPTR(PointsToRel ptr) {
        if (ptr == null) {
            return false;
        }
        Set<PointsToRel> ptrs = get(ptr.getPointerSymbol());
        return (ptrs != null && ptrs.contains(ptr));
    }

    /**
    * The Universe Domain is associated with the Points-To Domain 
    * and hence the points-to analyzer. We use this domain to 
    * represent the most conservative analysis result i.e. 
    * all pointers point-to all possible locations. In order 
    * to successfully implement forward data-flow analysis, 
    * a starting point and an end point are necessary to 
    * monotonically increase from one to the other. The 
    * Universe Domain represents the largest conservative 
    * assumption, points-to analysis cannot create an 
    * update to this Domain.
    * 
    * @see Domain
    * @see NullDomain
    * @see PointsToDomain
    * @see PointsToAnalysis
    */
    public static class Universe implements Domain {

        private static final Universe universe = new Universe();

        private Universe() {
            super();
        }

        public Domain union(Domain other) {
            return universe;
        }

        public Domain merge(Domain other) {
            return universe;
        }

        public Domain intersect(Domain other) {
            return universe;
        }

        public Domain diffStrong(Domain other) {
            return universe;
        }

        public Domain diffWeak(Domain other) {
            return universe;
        }

        public void kill(Set<Symbol> vars) {
            ;
        }

        public boolean equals(Domain other) {
            return (other == universe);
        }

        public static Universe getUniverse() {
            return universe;
        }

        @Override
        public Universe clone()
        {
            return universe;
        }
    
        @Override
        public String toString() {
            return "[POINTS-TO-UNIVERSE]";
        }
    }
}
