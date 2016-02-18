package cetus.hir;

import java.util.*;

/**
* Iterates over Traversable objects in depth-first order. The iteration starts
* from the root object that was specified in the constructor.
*/
public class DepthFirstIterator<E extends Traversable> extends IRIterator<E> {

    private Vector<Traversable> stack;

    // Prune set was replaced by prune list to avoid allocation of set iterator,
    // which requires extra memory; this is not a good idea since every IR
    // object needs to allocate such an iterator. By keeping this as an array
    // list there is no need for allocating iterator.
    private List<Class<? extends Traversable>> prune_list;

    /**
    * Creates a new iterator with the specified initial traversable object and
    * the optional pruned types.
    *
    * @param init The first object to visit.
    */
    public DepthFirstIterator(Traversable init) {
        super(init);
        stack = new Vector<Traversable>();
        stack.add(init);
        prune_list = new ArrayList<Class<? extends Traversable>>(4);
    }
    
    public boolean hasNext() {
        return !stack.isEmpty();
    }

    @SuppressWarnings("unchecked")
    public E next() {
        Traversable t = null;
        try {
            t = stack.remove(0);
        } catch(ArrayIndexOutOfBoundsException e) {
            // catching ArrayIndexOutofBoundsException, as remove method throws
            // this exception
            throw new NoSuchElementException();
        }
        if (t.getChildren() != null &&
            (prune_list.isEmpty() || !needsPruning(t.getClass()))) {
            List<Traversable> children = t.getChildren();
            // iterator or for-each statement will allocate a new object, which
            // is not memory-efficient. However this conventional style for
            // loop is efficient only for collection types that support random
            // access.
            for (int j = 0, i = 0; j < children.size(); j++) {
                Traversable child = children.get(j);
                if (child != null) {
                    stack.add(i++, child);
                }
            }
        }
        return (E)t;
    }

    private boolean needsPruning(Class<? extends Traversable> c) {
        for (int i = 0; i < prune_list.size(); i++) {
            if (prune_list.get(i).isAssignableFrom(c)) {
                return true;
            }
        }
        return false;
    }

    /**
    * Disables traversal from an object having the specified type. For example,
    * if traversal reaches an object with type <b>c</b>, it does not visit the
    * children of the object.
    *
    * @param c the object type to be pruned on.
    */
    public void pruneOn(Class<? extends E> c) {
        prune_list.add(c);
    }

    /**
    * Returns a linked list of objects of Class c in the IR
    *
    * @param c the object type to be collected.
    * @return the collected list.
    */
    @SuppressWarnings("unchecked")
    public <T extends Traversable> List<T> getList(Class<T> c) {
        List<T> ret = new ArrayList<T>();
        while (hasNext()) {
            Object o = next();
            if (c.isInstance(o)) {
                ret.add((T) o);
            }
        }
        return ret;
    }

    /**
    * Returns a set of objects of Class c in the IR
    *
    * @param c the object type to be collected.
    * @return the collected set.
    */
    @SuppressWarnings("unchecked")
    public <T extends Traversable> Set<T> getSet(Class<T> c) {
        HashSet<T> set = new HashSet<T>();
        while (hasNext()) {
            Object obj = next();
            if (c.isInstance(obj)) {
                set.add((T)obj);
            }
        }
        return set;
    }

    /**
    * Resets the iterator by setting the current position to the root object.
    * The pruned types are not cleared.
    */
    public void reset() {
        stack.clear();
        stack.add(root);
    }

    /**
    * Unlike the <b>reset</b> method, <b>clear</b> method also clears the
    * pruned types.
    */
    public void clear() {
        stack.clear();
        prune_list.clear();
        stack.add(root);
    }

}
