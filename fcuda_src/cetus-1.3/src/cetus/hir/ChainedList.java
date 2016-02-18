package cetus.hir;

import java.util.Collection;
import java.util.LinkedList;

/**
* Very similar to a LinkedList, except provides an <var>add</var> method that
* returns the list itself, allowing you to perform method chaining.  A
* beneficial side effect is that you can add multiple elements directly to a
* list created with the new operator, without first creating a new variable to
* hold a reference to the list object.  ChainedLists can be used anywhere a
* LinkedList or List occurs, and are entirely optional. It is possible to write
* passes for Cetus without using this class, however you will find that your
* code contains a lot of temporary lists created for the sole purpose of passing
* them as method arguments.
*/
public class ChainedList<T> extends LinkedList<T> {

    private static final long serialVersionUID = 3474L;

    /**
    * Constructs an empty list.
    */
    public ChainedList() {
        super();
    }

    /**
    * Constructs a list containing the elements of the specified collection,
    * in the order they are returned by the collection's iterator.
    *
    * @param c The collection whose elements are to be placed into this list.
    */
    public ChainedList(Collection<T> c) {
        super(c);
    }

    /**
    * Appends all of the elements in the specified collection to the end of this
    * list, in the order that they are returned by the specified collection's
    * iterator.
    *
    * @param c The collection whose elements are to be placed into this list.
    * @return this list.
    */
    public ChainedList addAllLinks(Collection<T> c) {
        addAll(c);
        return this;
    }

    /**
    * Appends the specified element to the end of this list.
    *
    * @param o Element to be appended to this list.
    * @return this list.
    */
    public ChainedList addLink(T o) {
        add(o);
        return this;
    }

}
