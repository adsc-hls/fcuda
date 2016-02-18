package cetus.analysis;

import cetus.hir.Symbol;

import java.util.Set;

/**
 * A class that implements Domain is used to represent container for data used
 * in data flow analysis. The following methods are common to most of the
 * data flow analysis.
 */
public interface Domain {

    /**
     * Returns union of the two domains if applicable.
     *
     * @param other the domain to be unioned.
     * @return the resulting Domain.
     */
    Domain union(Domain other);
  
    /**
     * Returns merge of the two domains (join operation).
     *
     * @param other the domain to be merged.
     * @return the resulting Domain.
     */
    Domain merge(Domain other);
  
    /**
     * Returns intersection of the two domains.
     *
     * @param other the domain to be intersected.
     * @return the resulting Domain.
     */
    Domain intersect(Domain other);
  
    /**
     * Returns the result of strong difference. Strong difference means
     * d1-d2 = {} if the difference is not computatble.
     * 
     * @param other the domain to be subtracted.
     * @return the resulting Domain.
     */
    Domain diffStrong(Domain other);
  
    /**
     * Returns the result of weak difference. Weak difference means
     * d1-d2 = d1 if the difference is not computable.
     *
     * @param other the domain to be subtracted.
     * @return the resulting Domain.
     */
    Domain diffWeak(Domain other);
  
    /**
     * Kills the data containing the specified set of symbols.
     *
     * @param vars the input set of symbols.
     */
    void kill(Set<Symbol> vars);
  
    /**
     * Checks if the other domain is equal to the current domain.
     *
     * @param other the domain to be compared.
     * @return true if they are equal, false otherwise.
     */
    boolean equals(Domain other);
  
    /**
    * Clones the domain.
    *
    * @return the cloned domain.
    */
    Domain clone();
}
