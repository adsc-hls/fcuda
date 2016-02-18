package cetus.analysis;

import cetus.hir.Symbol;

import java.util.Set;

/**
 * Class NullDomain represents a Domain that contains no data.
 * The meaning of a null domain depends on the type of the domain that are
 * used for a specific data flow problem. For example, a null domain means an
 * empty set for a domain with a "set" type whereas it represents the least
 * upper bound (TOP) for a lattice-based problem.
 */
public class NullDomain implements Domain {

    private static final NullDomain nulldomain = new NullDomain();

    public NullDomain() {
        ;
    }

    public Domain union(Domain other) {
        if (other == null || other instanceof NullDomain) {
            return this;
        } else {
            return other.union(this);
        }
    }

    public Domain merge(Domain other) {
        if (other == null || other instanceof NullDomain) {
            return this;
        } else {
            return other.merge(this);
        }
    }

    public Domain intersect(Domain other) {
        if (other == null || other instanceof NullDomain) {
            return this;
        } else {
            return other.intersect(this);
        }
    }

    public Domain diffStrong(Domain other) {
        return this;
    }

    public Domain diffWeak(Domain other) {
        return this;
    }

    public void kill(Set<Symbol> vars) {
        ;
    }

    public boolean equals(Domain other) {
        return (other != null && (other instanceof NullDomain));
    }

    public String toString() {
        return "[NULL]";
    }

    public static Domain getNull() {
        return nulldomain;
    }

    @Override
    public NullDomain clone() {
        return nulldomain;
    }
}
