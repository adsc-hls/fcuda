package cetus.hir;

import java.util.List;

/** 
* Any class implementing this interface can act
* as a tree node by providing access to its children
* and parent.
*/
public interface Traversable extends Printable {

    /**
    * Provides access to the children of this object as a list. This object is
    * free to internally implement its list as a LinkedList or an ArrayList.
    * It is generally not good practice to call this method yourself; the
    * ordering of children is not guaranteed so instead you should use the
    * methods of the particular class whose children you wish to access.
    *
    * @return the children as a list.
    */
    List<Traversable> getChildren();

    /**
    * Provides access to the parent of this object. Every IR object has at most
    * one parent.  (The parent relationship is the same as the parent
    * relationship in the parse tree, and is not to be confused with the base
    * class of a derived class declaration.)
    *
    * @return the parent of this object.
    */
    Traversable getParent();

    /**
    * Removes the specified child.
    *
    * @param child a reference to a child object that must match with ==.
    * @throws NotAChildException if the child does not exist.
    * @throws UnsupportedOperationException if the child exists but the
    *   parent refuses to let it go.  For instance, a BinaryExpression
    *   may refuse to remove its left side expression because that
    *   would lead to invalid IR.  The only way to change a child of such
    *   an expression is to replace it.
    */
    void removeChild(Traversable child);

    /**
    * Sets the <var>index</var><i>th</i> child of this object to <var>t</var>.
    * By calling this method, the old child located at the position <var>i</var>
    * will have a null parent.
    * It is generally not good practice to call this method yourself;
    * the ordering of children is not guaranteed so instead you should use the
    * methods of the particular class whose children you wish to access.
    * There are checks to prevent such actions as making a statement
    * a child of an expression, but you still may be able to do some damage
    * with this method if you are not careful.
    *
    * @throws NotAnOrphanException if <var>t</var> already has a parent.
    * @throws IllegalArgumentException if the type of the new child would
    *   violate a class invariant by becoming the <var>index</var><i>th</i>
    *   child.
    */
    void setChild(int index, Traversable t);

    /**
    * Sets the parent of this object.  Cetus checks that the
    * parent already considers this object a child.  The intent is to maintain
    * an ordering where first this object becomes a child of another object,
    * and then this object is told who its parent is.  Enforcing this order
    * makes it easier for Cetus to enforce other invariants and ensure the
    * IR is correct.
    *
    * @throws NotAChildException if the parent does not already consider
    *   this object a child.
    * @throws IllegalArgumentException (possibly) if the type of the parent
    *   would violate a class invariant.  Not all classes perform this check
    *   because, for example, Expressions can appear as children of so many
    *   things that it is simply not worth checking.
    */
    void setParent(Traversable t);

}
