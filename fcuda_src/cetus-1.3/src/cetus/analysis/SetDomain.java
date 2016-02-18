package cetus.analysis;

import cetus.hir.Symbol;
import cetus.hir.Tools;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Class SetDomain represents a set-based domain. For example, live variable and
 * may-modified variable can be solved using a set domain.
 * The semantics of a NullDomain with respect to a SetDomain is unknown set as
 * apposed to an empty set which is clearly represented using an empty Set
 * object.
 */
public class SetDomain<T> extends LinkedHashSet<T> implements Domain {

    private static final long serialVersionUID = 17L;
  
    /** Constructs an empty set domain */
    public SetDomain() {
        super();
    }

    /**
    * Constructs a new set domain with the specified set.
    * @param set the base set
    */
    public SetDomain(Set<T> set) {
        super();
        addAll(set);
    }

    public Domain union(Domain other) {
        if (other instanceof SetDomain) {
            SetDomain<T> ret = this.clone();
            Tools.addAll(ret, (SetDomain)other);
            return ret;
        } else {
            return NullDomain.getNull();
        }
    }

    public Domain merge(Domain other) {
        return union(other);
    }

    public Domain intersect(Domain other) {
        SetDomain<T> ret = this.clone();
        if (other instanceof SetDomain) {
            Tools.retainAll(ret, (SetDomain)other);
        } // intersection with other types of domain results in itself.
        return ret;
    }

    public Domain diffStrong(Domain other) {
        if (other instanceof SetDomain) {
            SetDomain<T> ret = this.clone();
            Tools.removeAll(ret, (SetDomain)other);
            return ret;
        } else {
            return NullDomain.getNull();
        }
    }

    public Domain diffWeak(Domain other) {
        SetDomain<T> ret = this.clone();
        if (other instanceof SetDomain) {
            Tools.removeAll(ret, (SetDomain)other);
        }
        return ret;
    }

    public void kill(Set<Symbol> vars) {
        ; // not supported, but need to be defined.
    }

    @Override
    public boolean equals(Domain other) {
        return super.equals(other);
    }

    @Override
    public SetDomain<T> clone() {
        SetDomain<T> ret = new SetDomain<T>();
        ret.addAll(this);
        return ret;
    }

    @Override
    public String toString() {
        return super.toString();
    }
}
